"""Tests for the Flask application and PDF reporting.

A stub inference service is injected so the routes can be exercised without
loading a real model or the FAISS index. That keeps the suite fast and makes it
runnable on a fresh clone.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import cv2
import pytest

from neuroscan.web.services import AnalysisResult, NotABrainScanError

from .conftest import synthetic_scan


class StubService:
    """Minimal stand-in for :class:`InferenceService`."""

    def __init__(self, cfg, *, ready: bool = True, llm: bool = True):
        self.cfg = cfg
        self._ready = ready
        self._llm = llm
        self.model_metadata = {
            "architecture": "efficientnet_b0",
            "class_names": ["normal", "abnormal"],
            "preprocessing": {"image_size": 224, "clahe_clip_limit": 2.0, "clahe_tile_grid": 8},
            "metrics": {"accuracy": 0.97, "recall": 0.98, "auc_roc": 0.99},
        }
        self.reject_next = False
        self.purge_calls = 0

    @property
    def status(self):
        return {"model": self._ready, "rag": True, "llm": self._llm,
                "llm_provider": "echo", "errors": [] if self._ready else ["no model"]}

    @property
    def is_ready(self):
        return self._ready

    def purge_old_uploads(self):
        self.purge_calls += 1
        return 0

    def analyse(self, image_path, *, analysis_id=None, check_is_brain=True):
        if self.reject_next:
            raise NotABrainScanError("not a brain scan")

        assets = self.cfg.paths.uploads_dir / analysis_id
        assets.mkdir(parents=True, exist_ok=True)
        image = synthetic_scan(lesion=True, seed=9)
        scan_path = assets / "scan.png"
        heat_path = assets / "heatmap.png"
        cv2.imwrite(str(scan_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(heat_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return AnalysisResult(
            analysis_id=analysis_id,
            prediction="abnormal",
            prediction_index=1,
            confidence=0.87,
            probabilities={"normal": 0.13, "abnormal": 0.87},
            threshold=0.5,
            architecture="efficientnet_b0",
            scan_image_url=f"/media/{analysis_id}/scan.png",
            heatmap_image_url=f"/media/{analysis_id}/heatmap.png",
            scan_image_path=scan_path,
            heatmap_image_path=heat_path,
            heatmap_focus_ratio=0.12,
            inference_ms=42.0,
            created_at=datetime.now(UTC),
        )

    def attach_advisory(self, result, language="en"):
        result.advisory_text = "## What this result means\n\nFeatures needing review."
        result.advisory_citations = ["Ring-Enhancing Lesion (reviewed 2026-08-13)"]
        result.red_flags = ["Sudden severe headache", "Seizure"]
        result.red_flag_instruction = "Go to hospital immediately if:"
        return result

    def ask(self, question, *, history=None, language=None, scan_prediction=None,
            scan_confidence=None):
        from neuroscan.chatbot.engine import ChatResponse
        from neuroscan.chatbot.language import detect_language

        resolved = language or detect_language(question)
        return ChatResponse(
            text="Possible causes include infection and tumour.",
            language=resolved,
            citations=["Tuberculoma (reviewed 2026-08-13)"],
            retrieved_count=3,
            provider="echo",
        )


@pytest.fixture
def app(cfg):
    from neuroscan.web.app import create_app

    application = create_app(cfg, service=StubService(cfg))
    application.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def upload_payload(name: str = "scan.png") -> dict:
    image = synthetic_scan(lesion=True, seed=4)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return {"scan": (io.BytesIO(buffer.tobytes()), name)}


class TestCsrf:
    """CSRF protection on the browser-facing routes.

    requirements.txt listed Flask-WTF "for CSRF protection on upload + chat
    forms" and nothing ever enabled it, so the claim was false and the
    endpoints were open. These tests run with protection ON, unlike the rest of
    the suite.
    """

    @pytest.fixture
    def csrf_client(self, cfg):
        from neuroscan.web.app import create_app

        application = create_app(cfg, service=StubService(cfg))
        application.config.update(TESTING=True, WTF_CSRF_ENABLED=True)
        return application.test_client()

    def test_upload_without_a_token_is_rejected(self, csrf_client):
        response = csrf_client.post(
            "/analyse", data=upload_payload(), content_type="multipart/form-data"
        )
        assert response.status_code == 400

    def test_chat_without_a_token_is_rejected(self, csrf_client):
        response = csrf_client.post("/chat", json={"question": "What is a glioma?"})
        assert response.status_code == 400

    def test_pages_expose_a_token(self, csrf_client):
        body = csrf_client.get("/").get_data(as_text=True)
        assert 'name="csrf-token"' in body
        assert 'name="csrf_token"' in body

    def test_programmatic_api_stays_exempt(self, csrf_client):
        """The JSON API is called by scripts holding no session cookie, so the
        attack CSRF prevents cannot apply to it."""
        response = csrf_client.post(
            "/api/analyse", data=upload_payload(), content_type="multipart/form-data"
        )
        assert response.status_code == 200


class TestPages:
    def test_index_loads(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Axial Screening Assistant" in response.data

    def test_index_shows_the_disclaimer(self, client):
        """Ethics 5.3 requires it on all outputs."""
        body = client.get("/").get_data(as_text=True)
        assert "not a diagnosis" in body.lower()

    def test_nepali_interface(self, client):
        body = client.get("/?lang=ne").get_data(as_text=True)
        assert "स्क्रिनिङ सहायक" in body
        assert 'lang="ne"' in body

    def test_language_choice_persists_in_session(self, client):
        client.get("/?lang=ne")
        assert 'lang="ne"' in client.get("/").get_data(as_text=True)

    def test_about_page_loads(self, client):
        assert client.get("/about").status_code == 200

    def test_help_page_loads_in_both_languages(self, client):
        body = client.get("/help").get_data(as_text=True)
        assert "What does Abnormal mean?" in body
        nepali = client.get("/help?lang=ne").get_data(as_text=True)
        assert "सहयोग" in nepali

    def test_landing_shows_the_two_classes_scannable(self, client):
        """The two trained classes must be visible at a glance, and the
        landing page must not imply the classifier names diseases."""
        body = client.get("/").get_data(as_text=True)
        assert "2 trained classes" in body
        assert "Does not identify a specific disease." in body

    def test_no_ai_marketing_language(self, client):
        body = client.get("/").get_data(as_text=True)
        for phrase in ("AI powered", "artificial intelligence", "Revolutionary", "Cutting-edge"):
            assert phrase not in body

    def test_navigation_marks_the_active_page(self, client):
        assert 'aria-current="page"' in client.get("/about").get_data(as_text=True)

    def test_health_endpoint(self, client):
        payload = client.get("/health").get_json()
        assert payload["ready"] is True
        assert "status" in payload

    def test_unknown_page_returns_404(self, client):
        assert client.get("/no-such-page").status_code == 404


class TestUpload:
    def test_valid_upload_redirects_to_result(self, client):
        response = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        assert response.status_code == 302
        assert "/result/" in response.headers["Location"]

    def test_result_page_shows_verdict_and_confidence(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "87.0%" in body
        assert "Grad-CAM" in body or "heatmap" in body.lower()

    def test_result_page_renders_advisory_as_html_not_markdown(self, client):
        """Regression: the page showed '## What this result means' literally.
        The stub advisory is markdown; the page must carry converted HTML."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "<h3>What this result means</h3>" in body
        assert "## What this result means" not in body

    def test_result_page_states_the_training_classes(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "trained on 2 classes" in body

    def test_verdict_is_a_sentence_not_a_stamp(self, client):
        """'Abnormal pattern detected' describes what the model did. A giant
        ABNORMAL reads as a diagnosis, which this system must never imply."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "Abnormal pattern detected" in body
        assert "not the probability" in body  # the confidence explainer

    def test_result_page_shows_scan_reference(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert f"NS-{analysis_id[:6].upper()}" in body

    def test_failed_heatmap_gets_a_statement_not_a_black_square(self, app, client):
        """A blank Grad-CAM is a known failure mode of the explanation step;
        the page must say no map was generated rather than show an empty
        overlay."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]
        app.config["NEUROSCAN_ANALYSES"][analysis_id].heatmap_failed = True
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "No attention map could be generated" in body
        assert "does not change the classification result" in body
        assert "heatmap.png" not in body

    def test_nepali_result_uses_devanagari_emergency_number(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"] + "?lang=ne").get_data(as_text=True)
        assert "१०२" in body
        client.get("/?lang=en")

    def test_chat_log_is_keyboard_scrollable(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert 'role="log"' in body
        log_tag = body.split('id="chat-log"')[0].rsplit("<div", 1)[1] + 'id="chat-log"' + \
            body.split('id="chat-log"')[1].split(">")[0]
        assert 'tabindex="0"' in log_tag

    def test_normal_result_never_shows_a_disease_list(self, app, client):
        """Template-level guard: even if bad data reaches the page, a normal
        verdict must not be followed by "Possible causes"."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]
        analysis = app.config["NEUROSCAN_ANALYSES"][analysis_id]
        analysis.prediction = "normal"
        analysis.advisory_structured = {
            "summary": "The scan was classified normal.",
            "possible_causes": [{"name": "Neurocysticercosis", "note": "Should never render."}],
            "next_steps": ["Review symptoms with a clinician."],
        }
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "No abnormal pattern detected" in body
        assert "Possible causes" not in body
        assert "Neurocysticercosis" not in body

    def test_no_raw_markdown_or_separator_leaks(self, client):
        """No ##, ** or --- may reach the visible page."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "##" not in body
        assert "**" not in body
        assert "\n---" not in body

    def test_result_separates_classifier_from_advisory(self, client):
        """The page must never let a reader believe the CNN named a disease."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "Model result" in body
        assert "Clinical context" in body
        assert "only separates normal from abnormal" in body

    def test_structured_advisory_renders_natively(self, app, client):
        """When the model returned valid JSON, the page renders the structure
        itself rather than converted markdown."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]
        analysis = app.config["NEUROSCAN_ANALYSES"][analysis_id]
        analysis.advisory_structured = {
            "summary": "The scan was flagged as abnormal and needs review.",
            "possible_causes": [
                {"name": "Neurocysticercosis", "note": "Common and treatable in Nepal."},
            ],
            "next_steps": ["Arrange review by a physician."],
        }
        body = client.get(redirect.headers["Location"]).get_data(as_text=True)
        assert "Neurocysticercosis" in body
        assert "Possible causes" in body
        assert "Arrange review by a physician." in body
        # The markdown fallback must not render alongside the structured form.
        assert "<h3>What this result means</h3>" not in body

    def test_missing_file_is_rejected(self, client):
        response = client.post("/analyse", data={}, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_disallowed_extension_is_rejected(self, client):
        payload = {"scan": (io.BytesIO(b"not an image"), "malware.exe")}
        response = client.post("/analyse", data=payload, content_type="multipart/form-data")
        assert response.status_code == 400
        assert b"Unsupported file type" in response.data

    def test_non_brain_image_is_rejected(self, app, client):
        """Feeding a photograph to the model yields a confident, meaningless
        prediction, which is worse than a clear rejection."""
        app.config["NEUROSCAN_SERVICE"].reject_next = True
        response = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        assert response.status_code == 400
        assert b"does not appear to be a brain MRI" in response.data

    def test_upload_triggers_retention_purge(self, app, client):
        client.post("/analyse", data=upload_payload(), content_type="multipart/form-data")
        assert app.config["NEUROSCAN_SERVICE"].purge_calls >= 1

    def test_oversized_upload_returns_413(self, app, client):
        app.config["MAX_CONTENT_LENGTH"] = 512
        payload = {"scan": (io.BytesIO(b"x" * 4096), "big.png")}
        response = client.post("/analyse", data=payload, content_type="multipart/form-data")
        assert response.status_code == 413


class TestMedia:
    def test_serves_generated_images(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]
        response = client.get(f"/media/{analysis_id}/scan.png")
        assert response.status_code == 200
        assert response.data[:4] == b"\x89PNG"

    @pytest.mark.parametrize(
        "analysis_id", ["../../../etc", "..%2f..%2fsecret", "a/../../b", "!!!"]
    )
    def test_rejects_path_traversal(self, client, analysis_id):
        assert client.get(f"/media/{analysis_id}/scan.png").status_code in (404, 308, 400)


class TestChat:
    def test_answers_a_question(self, client):
        response = client.post("/chat", json={"question": "What does abnormal mean?"})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["answer"]
        assert payload["language"] == "en"

    def test_detects_nepali(self, client):
        payload = client.post("/chat", json={"question": "यो नतिजाको अर्थ के हो?"}).get_json()
        assert payload["language"] == "ne"

    def test_empty_question_rejected(self, client):
        assert client.post("/chat", json={"question": "   "}).status_code == 400

    def test_history_is_kept_in_session(self, client):
        client.post("/chat", json={"question": "First question?"})
        client.post("/chat", json={"question": "Second question?"})
        with client.session_transaction() as session:
            assert len(session["chat_history"]) == 2

    def test_response_carries_the_disclaimer_separately(self, client):
        """Answers no longer embed the disclaimer in their prose (the page
        shows one persistently); programmatic consumers get it as a field."""
        payload = client.post("/chat", json={"question": "What is a glioma?"}).get_json()
        assert payload["disclaimer"]
        assert "diagnosis" in payload["disclaimer"].lower()

    def test_form_post_degrades_to_a_redirect(self, client):
        """With JavaScript off the question form is a real POST; the answer
        arrives by redirecting back to the server-rendered transcript, not as
        raw JSON in the browser."""
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]

        response = client.post("/chat", data={
            "question": "What does this result mean?",
            "analysis_id": analysis_id,
            "language": "en",
        })
        assert response.status_code == 302
        assert f"/result/{analysis_id}" in response.headers["Location"]
        assert "#questions" in response.headers["Location"]

        with client.session_transaction() as session:
            assert len(session["chat_history"]) == 1

        body = client.get(f"/result/{analysis_id}").get_data(as_text=True)
        assert "What does this result mean?" in body

    def test_form_post_with_empty_question_redirects_not_400(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]
        response = client.post("/chat", data={"question": "  ", "analysis_id": analysis_id})
        assert response.status_code == 302


class TestHistory:
    def test_empty_history_page_loads(self, client):
        body = client.get("/history").get_data(as_text=True)
        assert "No previous scans" in body

    def test_lists_this_sessions_scans_newest_first(self, client):
        first = client.post("/analyse", data=upload_payload(),
                            content_type="multipart/form-data")
        second = client.post("/analyse", data=upload_payload(),
                             content_type="multipart/form-data")
        first_id = first.headers["Location"].rstrip("/").split("/")[-1]
        second_id = second.headers["Location"].rstrip("/").split("/")[-1]

        body = client.get("/history").get_data(as_text=True)
        first_ref = f"NS-{first_id[:6].upper()}"
        second_ref = f"NS-{second_id[:6].upper()}"
        assert first_ref in body
        assert second_ref in body
        assert body.index(second_ref) < body.index(first_ref)

    def test_history_is_session_scoped(self, app, client):
        """Another browser session must not see this session's scans."""
        client.post("/analyse", data=upload_payload(), content_type="multipart/form-data")
        other = app.test_client()
        body = other.get("/history").get_data(as_text=True)
        assert "NS-" not in body


class TestApi:
    def test_returns_json_result(self, client):
        response = client.post("/api/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["prediction"] == "abnormal"
        assert 0.0 <= payload["confidence"] <= 1.0
        assert "analysis_id" in payload

    def test_rejects_missing_file(self, client):
        response = client.post("/api/analyse", data={}, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_non_brain_returns_422(self, app, client):
        app.config["NEUROSCAN_SERVICE"].reject_next = True
        response = client.post("/api/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        assert response.status_code == 422


class TestDegradedMode:
    def test_app_starts_without_a_model(self, cfg):
        """A clinic machine should show a clear message, not fail to start."""
        from neuroscan.web.app import create_app

        application = create_app(cfg, service=StubService(cfg, ready=False))
        application.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
        client = application.test_client()

        assert client.get("/").status_code == 200
        assert b"No trained model" in client.get("/").data
        assert client.post("/analyse", data=upload_payload(),
                           content_type="multipart/form-data").status_code == 503

    def test_warns_when_llm_unavailable(self, cfg):
        from neuroscan.web.app import create_app

        application = create_app(cfg, service=StubService(cfg, llm=False))
        application.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
        body = application.test_client().get("/").get_data(as_text=True)
        assert "language model is unavailable" in body.lower()


class TestPdfReport:
    def test_generates_a_valid_pdf(self, cfg, tmp_path):
        from neuroscan.reporting.pdf_report import ReportData, build_report

        image_path = tmp_path / "scan.png"
        cv2.imwrite(str(image_path),
                    cv2.cvtColor(synthetic_scan(lesion=True, seed=6), cv2.COLOR_RGB2BGR))

        out = build_report(
            ReportData(
                prediction="abnormal",
                confidence=0.87,
                language="en",
                scan_image_path=image_path,
                heatmap_image_path=image_path,
                advisory_text="## What this means\n\nFeatures requiring review.\n\n- One\n- Two",
                citations=["Tuberculoma (reviewed 2026-08-13)"],
                report_id="testreport",
            ),
            tmp_path / "report.pdf",
        )

        assert out.exists()
        assert out.read_bytes()[:5] == b"%PDF-"
        assert out.stat().st_size > 3000

    def test_nepali_report_generates(self, tmp_path):
        """Devanagari is rendered through Pillow and embedded as images,
        because ReportLab performs no complex text shaping."""
        from neuroscan.reporting.pdf_report import ReportData, build_report

        out = build_report(
            ReportData(
                prediction="असामान्य",
                confidence=0.8,
                language="ne",
                advisory_text="## नतिजा\n\nयो निदान होइन।",
                report_id="nepali",
            ),
            tmp_path / "report_ne.pdf",
        )
        assert out.exists()
        assert out.read_bytes()[:5] == b"%PDF-"

    def test_report_route_downloads(self, client):
        redirect = client.post("/analyse", data=upload_payload(),
                               content_type="multipart/form-data")
        analysis_id = redirect.headers["Location"].rstrip("/").split("/")[-1]
        response = client.get(f"/report/{analysis_id}")
        assert response.status_code == 200
        assert response.mimetype == "application/pdf"
        assert response.data[:5] == b"%PDF-"

    def test_report_for_unknown_analysis_returns_404(self, client):
        assert client.get("/report/deadbeef1234").status_code == 404
