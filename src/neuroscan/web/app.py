"""Flask application factory and routes.

Session state holds only an analysis id and the chat transcript - never the
image, and never anything patient-identifying. Uploaded scans live on disk
under a random id and are purged on a retention timer.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

from neuroscan.chatbot.language import get_ui_strings
from neuroscan.safety import get_disclaimer, get_red_flags
from neuroscan.textfmt import drop_leading_title, to_safe_html
from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config

log = get_logger("web.app")

ALLOWED_MIME_PREFIXES = ("image/",)


def _resolve_language(cfg: Config) -> str:
    """Resolve the interface language for this request.

    Order: explicit query parameter, then session, then configured default.
    The query parameter also persists the choice, so the language toggle works
    with a plain link and needs no JavaScript - which matters on the low-end
    devices this is meant to run on.
    """
    requested = request.args.get("lang")
    if requested in {"en", "ne"}:
        session["language"] = requested
        return requested
    stored = session.get("language")
    if stored in {"en", "ne"}:
        return stored
    return cfg.chatbot.default_language


def create_app(cfg: Config | None = None, *, service=None) -> Flask:
    """Build the Flask application.

    Args:
        cfg: Configuration. Loaded from defaults when omitted.
        service: Pre-built :class:`InferenceService`. Injected by tests to
            avoid loading real models.
    """
    from neuroscan.config import load_config
    from neuroscan.web.services import (
        InferenceService,
        ModelUnavailableError,
        NotABrainScanError,
        ServiceError,
    )

    cfg = cfg or load_config()
    cfg.paths.ensure_all()

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    import os

    secret = os.environ.get(cfg.web.secret_key_env_var)
    if not secret:
        # Ephemeral key: sessions do not survive a restart, which is the safe
        # default. A deployment that needs persistent sessions sets the
        # environment variable.
        secret = secrets.token_hex(32)
        log.warning(
            "%s is not set - using an ephemeral session key. Sessions will not "
            "survive a restart.", cfg.web.secret_key_env_var,
        )

    app.config.update(
        SECRET_KEY=secret,
        MAX_CONTENT_LENGTH=cfg.web.max_upload_mb * 1024 * 1024,
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=cfg.web.session_lifetime_minutes),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        JSON_AS_ASCII=False,
        TEMPLATES_AUTO_RELOAD=cfg.web.debug,
    )

    # CSRF protection on every state-changing route. requirements.txt listed
    # Flask-WTF for this and nothing ever enabled it, so the claim was false
    # and the endpoints were open. The risk is small while bound to localhost
    # and real the moment anyone runs with --host 0.0.0.0, which the CLI
    # invites.
    csrf = CSRFProtect()
    csrf.init_app(app)
    app.config["NEUROSCAN_CSRF"] = csrf

    app.config["NEUROSCAN_CFG"] = cfg
    app.config["NEUROSCAN_SERVICE"] = service or InferenceService(cfg)

    # In-process store of completed analyses. A single-process offline
    # deployment does not warrant a database, and persisting scan results would
    # contradict the retention commitment in docs/ETHICS.md.
    app.config["NEUROSCAN_ANALYSES"] = {}

    def get_service():
        return app.config["NEUROSCAN_SERVICE"]

    def base_context(language: str) -> dict:
        instruction, red_flags = get_red_flags(language)  # type: ignore[arg-type]
        return {
            "t": get_ui_strings(language),  # type: ignore[arg-type]
            "language": language,
            "other_language": "ne" if language == "en" else "en",
            "disclaimer": get_disclaimer(language),  # type: ignore[arg-type]
            "disclaimer_short": get_disclaimer(language, short=True),  # type: ignore[arg-type]
            "max_upload_bytes": cfg.web.max_upload_mb * 1024 * 1024,
            "max_upload_mb": cfg.web.max_upload_mb,
            "emergency_instruction": instruction,
            "red_flags": red_flags,
            "status": get_service().status,
        }

    # ------------------------------------------------------------------ routes

    @app.route("/")
    def index():
        language = _resolve_language(cfg)
        return render_template("index.html", **base_context(language))

    @app.route("/health")
    def health():
        """Readiness probe, also used by the UI to warn about degraded parts."""
        service = get_service()
        return jsonify({
            "ready": service.is_ready,
            "status": service.status,
            "version": __import__("neuroscan").__version__,
        })

    @app.route("/analyse", methods=["POST"])
    def analyse():
        language = _resolve_language(cfg)
        service = get_service()
        strings = get_ui_strings(language)  # type: ignore[arg-type]

        def fail(message_key: str, detail: str = "", code: int = 400):
            return render_template(
                "index.html",
                error=strings.get(message_key, message_key),
                error_detail=detail,
                **base_context(language),
            ), code

        if not service.is_ready:
            return fail("error_model_missing", "; ".join(service.status.get("errors", [])), 503)

        uploaded = request.files.get("scan")
        if uploaded is None or not uploaded.filename:
            return fail("error_no_file")

        filename = secure_filename(uploaded.filename)
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in cfg.web.allowed_extensions:
            return fail(
                "error_bad_type",
                f"Received '.{suffix}'. Allowed: {', '.join(cfg.web.allowed_extensions)}.",
            )
        if uploaded.mimetype and not uploaded.mimetype.startswith(ALLOWED_MIME_PREFIXES):
            return fail("error_bad_type", f"Content type '{uploaded.mimetype}' is not an image.")

        analysis_id = uuid.uuid4().hex[:12]
        upload_dir = cfg.paths.uploads_dir / analysis_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        source_path = upload_dir / f"original.{suffix}"
        uploaded.save(source_path)

        # Opportunistic cleanup, so an always-on clinic machine does not
        # accumulate scans indefinitely.
        service.purge_old_uploads()

        try:
            result = service.analyse(source_path, analysis_id=analysis_id)
        except NotABrainScanError:
            return fail("error_not_brain")
        except ModelUnavailableError as exc:
            return fail("error_model_missing", str(exc), 503)
        except Exception as exc:
            log.exception("Analysis failed for %s", analysis_id)
            return fail("error_bad_type", str(exc), 500)

        result = service.attach_advisory(result, language=language)  # type: ignore[arg-type]

        app.config["NEUROSCAN_ANALYSES"][analysis_id] = result
        session["analysis_id"] = analysis_id
        session["chat_history"] = []
        # Session-scoped history only: results live in process memory and
        # uploads purge on a timer, so "previous scans" deliberately means
        # this browser session, not a patient database (docs/ETHICS.md).
        history_ids = session.get("history_ids", [])
        history_ids.append(analysis_id)
        session["history_ids"] = history_ids[-25:]
        session.permanent = True

        return redirect(url_for("result", analysis_id=analysis_id))

    @app.route("/result/<analysis_id>")
    def result(analysis_id: str):
        language = _resolve_language(cfg)
        analysis = app.config["NEUROSCAN_ANALYSES"].get(analysis_id)
        if analysis is None:
            abort(404)

        # The advisory is stored as markdown (the PDF path needs it that way);
        # the page needs HTML, converted with escape-first sanitisation because
        # this is model output. Rendering the raw text put every ** and ## the
        # model emitted literally on screen.
        advisory_html = to_safe_html(
            drop_leading_title(analysis.advisory_text, "Clinical Advisory")
        )
        return render_template(
            "result.html",
            result=analysis,
            advisory_html=advisory_html,
            chat_history=session.get("chat_history", []),
            **base_context(language),
        )

    @app.route("/media/<analysis_id>/<path:filename>")
    def media(analysis_id: str, filename: str):
        """Serve generated images.

        ``analysis_id`` is validated as hex so it cannot escape the uploads
        directory, and ``send_from_directory`` blocks traversal in the
        filename.
        """
        if not analysis_id.isalnum() or len(analysis_id) > 32:
            abort(404)
        directory = cfg.paths.uploads_dir / analysis_id
        if not directory.is_dir():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/chat", methods=["POST"])
    def chat():
        language_override = None
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or request.form.get("question") or "").strip()

        # A plain form submission means JavaScript is off. The answer then
        # arrives by redirecting back to the result page, where the session
        # transcript is server-rendered - raw JSON in the browser is not a
        # fallback, it is a dead end.
        wants_html = not request.is_json
        analysis_id = (
            payload.get("analysis_id")
            or request.form.get("analysis_id")
            or session.get("analysis_id")
        )
        analysis = app.config["NEUROSCAN_ANALYSES"].get(analysis_id) if analysis_id else None

        def back(failed: bool = False):
            if analysis is not None:
                target = url_for("result", analysis_id=analysis_id, chat_error=1 if failed else None)
                return redirect(f"{target}#questions")
            return redirect(url_for("index"))

        if not question:
            if wants_html:
                return back()
            return jsonify({"error": "empty_question"}), 400

        requested = payload.get("language") or request.form.get("language")
        if requested in {"en", "ne"}:
            language_override = requested

        service = get_service()
        history = session.get("chat_history", [])

        try:
            response = service.ask(
                question,
                history=history,
                language=language_override,
                scan_prediction=analysis.prediction if analysis else None,
                scan_confidence=analysis.confidence if analysis else None,
            )
        except ServiceError as exc:
            if wants_html:
                return back(failed=True)
            return jsonify({"error": "unavailable", "detail": str(exc)}), 503
        except Exception as exc:
            log.exception("Chat request failed")
            if wants_html:
                return back(failed=True)
            return jsonify({"error": "internal", "detail": str(exc)}), 500

        history.append({"question": question, "answer": response.text})
        session["chat_history"] = history[-cfg.chatbot.max_history_turns * 2 :]

        if wants_html:
            return back()

        return jsonify({
            "answer": response.text,
            "language": response.language,
            "citations": response.citations,
            "refused": response.refused,
            "degraded": response.degraded,
            "retrieved": response.retrieved_count,
            # Answers no longer embed the disclaimer in their prose (the page
            # shows one persistently). Programmatic consumers get it here.
            "disclaimer": get_disclaimer(response.language),
        })

    @app.route("/report/<analysis_id>")
    def report(analysis_id: str):
        language = _resolve_language(cfg)
        analysis = app.config["NEUROSCAN_ANALYSES"].get(analysis_id)
        if analysis is None:
            abort(404)

        from neuroscan.reporting.pdf_report import ReportData, build_report

        out_path = cfg.paths.reports_dir / f"neuroscan_report_{analysis_id}.pdf"
        data = ReportData(
            prediction=analysis.prediction,
            confidence=analysis.confidence,
            language=language,  # type: ignore[arg-type]
            architecture=analysis.architecture,
            threshold=analysis.threshold,
            scan_image_path=analysis.scan_image_path,
            heatmap_image_path=analysis.heatmap_image_path,
            advisory_text=analysis.advisory_text,
            citations=analysis.advisory_citations,
            heatmap_is_diffuse=analysis.heatmap_is_diffuse,
            chat_history=session.get("chat_history", []),
            report_id=analysis_id,
            generated_at=analysis.created_at,
            degraded=analysis.advisory_degraded,
        )

        try:
            build_report(data, out_path)
        except Exception as exc:
            log.exception("Report generation failed for %s", analysis_id)
            abort(500, description=f"Could not generate the report: {exc}")

        return send_file(
            out_path,
            as_attachment=True,
            download_name=f"neuroscan_report_{analysis_id}.pdf",
            mimetype="application/pdf",
        )

    @app.route("/api/analyse", methods=["POST"])
    @csrf.exempt
    def api_analyse():
        """JSON endpoint, for the usability study scripts and integration tests.

        CSRF-exempt because it is a programmatic API called by scripts that
        hold no session cookie - the attack CSRF prevents requires ambient
        credentials, which this endpoint never uses. The browser-facing routes
        are protected.
        """
        service = get_service()
        if not service.is_ready:
            return jsonify({"error": "model_unavailable", "status": service.status}), 503

        uploaded = request.files.get("scan")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "no_file"}), 400

        language = request.form.get("language", cfg.chatbot.default_language)
        analysis_id = uuid.uuid4().hex[:12]
        upload_dir = cfg.paths.uploads_dir / analysis_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(secure_filename(uploaded.filename)).suffix.lower().lstrip(".") or "png"
        if suffix not in cfg.web.allowed_extensions:
            return jsonify({"error": "bad_type", "allowed": cfg.web.allowed_extensions}), 400

        source_path = upload_dir / f"original.{suffix}"
        uploaded.save(source_path)

        try:
            result = service.analyse(source_path, analysis_id=analysis_id)
        except NotABrainScanError as exc:
            return jsonify({"error": "not_a_brain_scan", "detail": str(exc)}), 422
        except Exception as exc:
            log.exception("API analysis failed")
            return jsonify({"error": "internal", "detail": str(exc)}), 500

        if request.form.get("advisory", "true").lower() != "false":
            result = service.attach_advisory(result, language=language)  # type: ignore[arg-type]

        app.config["NEUROSCAN_ANALYSES"][analysis_id] = result
        return jsonify(result.to_dict())

    @app.route("/history")
    def history():
        """This session's scans. Deliberately not a patient database.

        Results live only in process memory and uploaded images purge on a
        timer, so history is scoped to the browser session and vanishes on
        restart. That is the retention policy working, not a limitation.
        """
        language = _resolve_language(cfg)
        analyses = app.config["NEUROSCAN_ANALYSES"]
        entries = [
            analyses[analysis_id]
            for analysis_id in reversed(session.get("history_ids", []))
            if analysis_id in analyses
        ]
        return render_template("history.html", entries=entries, **base_context(language))

    @app.route("/help")
    def help_page():
        language = _resolve_language(cfg)
        return render_template("help.html", **base_context(language))

    @app.route("/about")
    def about():
        language = _resolve_language(cfg)
        service = get_service()
        return render_template(
            "about.html",
            model_metadata=service.model_metadata,
            cv_stats=getattr(service, "cv_stats", None),
            **base_context(language),
        )

    # ------------------------------------------------------------- error pages

    @app.errorhandler(404)
    def not_found(_error):
        language = _resolve_language(cfg)
        return render_template("error.html", code=404,
                               message="Page not found", **base_context(language)), 404

    @app.errorhandler(413)
    def too_large(_error):
        language = _resolve_language(cfg)
        strings = get_ui_strings(language)  # type: ignore[arg-type]
        return render_template(
            "error.html", code=413,
            message=f"{strings['error_too_large']} Maximum {cfg.web.max_upload_mb} MB.",
            **base_context(language),
        ), 413

    @app.errorhandler(500)
    def server_error(error):
        language = _resolve_language(cfg)
        log.error("Internal server error: %s", error)
        return render_template("error.html", code=500,
                               message="An internal error occurred",
                               **base_context(language)), 500

    log.info("Flask application ready (debug=%s)", cfg.web.debug)
    return app


__all__ = ["create_app"]
