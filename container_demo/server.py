from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


APP_ROOT = Path(__file__).resolve().parents[1]
PREMIUM_ROOT = APP_ROOT / "runtime_models" / "premium"
HF_SCRIPTS = PREMIUM_ROOT / "scripts" / "hf_encoder"
CATT_DIR = PREMIUM_ROOT / "external" / "catt_punct_v4"
EO_POLISHED_CHECKPOINT = CATT_DIR / "catt_eo_ft_polished_v2" / "catt_eo_ft_best.pt"
EO_LEGACY_CHECKPOINT = CATT_DIR / "catt_eo_ft_balanced_v4_punct" / "catt_eo_ft_best.pt"


def resolve_runtime_path(env_name: str, default: Path, fallback: Path | None = None) -> Path:
    value = os.getenv(env_name, "").strip()
    candidate = Path(value) if value else default
    if value and not candidate.is_absolute():
        candidate = APP_ROOT / candidate
    if candidate.exists():
        return candidate
    if fallback is not None and fallback.exists():
        return fallback
    return candidate


def display_runtime_path(path: Path) -> str:
    try:
        return path.relative_to(APP_ROOT).as_posix()
    except ValueError:
        return str(path)


EO_CHECKPOINT = resolve_runtime_path("TASHKEEL_EO_CHECKPOINT", EO_POLISHED_CHECKPOINT, EO_LEGACY_CHECKPOINT)
EO_MODEL_LABEL = os.getenv("TASHKEEL_EO_LABEL", "Sukoon")
PREMIUM_CATT_CHECKPOINT = resolve_runtime_path("TASHKEEL_PREMIUM_CATT_CHECKPOINT", EO_LEGACY_CHECKPOINT)
HF_CHECKPOINT = PREMIUM_ROOT / "runs" / "aramodernbert_decomp_transformer_head_v1_cont_lr5e6_bs64_len512" / "last"
DUAL_STREAM_HEAD = (
    PREMIUM_ROOT
    / "runs"
    / "premium_dual_stream_head_v1_balanced_teacher_lr5e6_hard15k_tw005_pw005_20260501_172507"
    / "last_dual_stream_head.pt"
)
SPECIALIST_CKPT = (
    PREMIUM_ROOT / "runs" / "final_letter_specialist_v1" / "final_letter_specialist_best.pt"
)
PREMIUM_CORRECTION_SCALE = float(os.getenv("TASHKEEL_CORRECTION_SCALE", "0.0400"))
SPECIALIST_THRESHOLD = float(os.getenv("TASHKEEL_SPECIALIST_THRESHOLD", "0.40"))
WEB_ROOT = Path(__file__).resolve().parent / "webui" / "dist"
HTML_PATH = WEB_ROOT / "index.html"
FALLBACK_HTML_PATH = Path(__file__).resolve().with_name("index.html")
CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}
TASHKEEL_RE = re.compile(r"[\u064b-\u0652\u0670]")
ARABIC_BASE_RE = re.compile(r"[\u0621-\u064a\u0671]")
PAUSE_PUNCTUATION_RE = re.compile(r"([\u0621-\u064a\u0671])([\u064b-\u0652\u0670]+)(?=[.!?\u061f\u061b;:])")
TASHKEEL_CHARS = set(chr(code) for code in range(0x064B, 0x0653)) | {"\u0670"}
# Alif variants \u2014 hamza placement is spelling, not diacritization; never change these.
ALIF_FORMS = {"\u0627", "\u0623", "\u0625", "\u0622", "\u0671"}
SHADDA = "\u0651"
SUPERSCRIPT_ALIF = "\u0670"
VOWELS = ["\u064b", "\u064c", "\u064d", "\u064e", "\u064f", "\u0650", "\u0652"]
PAUSE_STRIP_MARKS = set(VOWELS)
WORD_COUNT_DEMO_SEGMENTS = [
    {
        "plain": "في الصباح كتب الطالب الدرس ثم قرأ صفحة من الكتاب، وسأل المعلم عن معنى الكلمة، فابتسم الطالب وقال إن العلم نور يهدي الناس إلى الطريق.",
        "gold": "فِي الصَّبَاحِ كَتَبَ الطَّالِبُ الدَّرْسَ ثُمَّ قَرَأَ صَفْحَةً مِنَ الْكِتَابِ، وَسَأَلَ الْمُعَلِّمَ عَنْ مَعْنَى الْكَلِمَةِ، فَابْتَسَمَ الطَّالِبُ وَقَالَ إِنَّ الْعِلْمَ نُورٌ يَهْدِي النَّاسَ إِلَى الطَّرِيقِ.",
    },
    {
        "plain": "عندما عاد المسافر إلى القرية، وجد الأطفال يلعبون قرب الساحة، ورأى النساء يخبزن الخبز، فجلس مع أبيه يحدثه عن رحلته الطويلة في المدن البعيدة كثيرا.",
        "gold": "عِنْدَمَا عَادَ الْمُسَافِرُ إِلَى الْقَرْيَةِ، وَجَدَ الْأَطْفَالَ يَلْعَبُونَ قُرْبَ السَّاحَةِ، وَرَأَى النِّسَاءَ يَخْبِزْنَ الْخُبْزَ، فَجَلَسَ مَعَ أَبِيهِ يُحَدِّثُهُ عَنْ رِحْلَتِهِ الطَّوِيلَةِ فِي الْمُدُنِ الْبَعِيدَةِ كَثِيرًا.",
    },
    {
        "plain": "اجتمع الفريق في المختبر بعد الظهر، وناقش الباحثون نتائج التجربة، ثم سجلوا الملاحظات بدقة، لأن العمل المنظم يساعد على فهم المشكلة بسرعة واتخاذ القرار المناسب.",
        "gold": "اِجْتَمَعَ الْفَرِيقُ فِي الْمُخْتَبَرِ بَعْدَ الظُّهْرِ، وَنَاقَشَ الْبَاحِثُونَ نَتَائِجَ التَّجْرِبَةِ، ثُمَّ سَجَّلُوا الْمُلَاحَظَاتِ بِدِقَّةٍ، لِأَنَّ الْعَمَلَ الْمُنَظَّمَ يُسَاعِدُ عَلَى فَهْمِ الْمُشْكِلَةِ بِسُرْعَةٍ وَاتِّخَاذِ الْقَرَارِ الْمُنَاسِبِ.",
    },
    {
        "plain": "في السوق القديم اشترت الأم فاكهة طازجة وخضارا كثيرة، ثم عادت إلى البيت لتعد طعاما لذيذا يجمع الأسرة حول المائدة مساء بفرح وهدوء ومحبة كبيرة.",
        "gold": "فِي السُّوقِ الْقَدِيمِ اِشْتَرَتِ الْأُمُّ فَاكِهَةً طَازَجَةً وَخُضَارًا كَثِيرَةً، ثُمَّ عَادَتْ إِلَى الْبَيْتِ لِتُعِدَّ طَعَامًا لَذِيذًا يَجْمَعُ الْأُسْرَةَ حَوْلَ الْمَائِدَةِ مَسَاءً بِفَرَحٍ وَهُدُوءٍ وَمَحَبَّةٍ كَبِيرَةٍ.",
    },
    {
        "plain": "قبل بدء الحصة رتب المعلم المقاعد، وكتب السؤال على اللوح، ثم طلب من الطلاب قراءة النص بصوت واضح واحترام آراء الزملاء أثناء النقاش في الصف.",
        "gold": "قَبْلَ بَدْءِ الْحِصَّةِ رَتَّبَ الْمُعَلِّمُ الْمَقَاعِدَ، وَكَتَبَ السُّؤَالَ عَلَى اللَّوْحِ، ثُمَّ طَلَبَ مِنَ الطُّلَّابِ قِرَاءَةَ النَّصِّ بِصَوْتٍ وَاضِحٍ وَاحْتِرَامَ آرَاءِ الزُّمَلَاءِ أَثْنَاءَ النِّقَاشِ فِي الصَّفِّ.",
    },
    {
        "plain": "سارت الطفلة مع جدتها في الحديقة، وشاهدت الأزهار الملونة، وسمعت تغريد العصافير، فقالت إن الجمال يسكن التفاصيل الصغيرة كل يوم حين ننظر إليه بعين شاكرة.",
        "gold": "سَارَتِ الطِّفْلَةُ مَعَ جَدَّتِهَا فِي الْحَدِيقَةِ، وَشَاهَدَتِ الْأَزْهَارَ الْمُلَوَّنَةَ، وَسَمِعَتْ تَغْرِيدَ الْعَصَافِيرِ، فَقَالَتْ إِنَّ الْجَمَالَ يَسْكُنُ التَّفَاصِيلَ الصَّغِيرَةَ كُلَّ يَوْمٍ حِينَ نَنْظُرُ إِلَيْهِ بِعَيْنٍ شَاكِرَةٍ.",
    },
    {
        "plain": "أعلن المدير خطة جديدة لتنظيم العمل، فاستمع الموظفون بعناية، واقترحوا أفكارا عملية، ثم اتفق الجميع على جدول واضح للمراجعة الأسبوعية وتحسين الخدمة في المرحلة القادمة.",
        "gold": "أَعْلَنَ الْمُدِيرُ خُطَّةً جَدِيدَةً لِتَنْظِيمِ الْعَمَلِ، فَاسْتَمَعَ الْمُوَظَّفُونَ بِعِنَايَةٍ، وَاقْتَرَحُوا أَفْكَارًا عَمَلِيَّةً، ثُمَّ اتَّفَقَ الْجَمِيعُ عَلَى جَدْوَلٍ وَاضِحٍ لِلْمُرَاجَعَةِ الْأُسْبُوعِيَّةِ وَتَحْسِينِ الْخِدْمَةِ فِي الْمَرْحَلَةِ الْقَادِمَةِ.",
    },
    {
        "plain": "زار الطبيب المدرسة ليشرح للطلاب أهمية النظافة، وتحدث عن غسل اليدين، وشرب الماء، والنوم المبكر، وممارسة الرياضة للحفاظ على صحة قوية وجسم نشيط وعقل يقظ.",
        "gold": "زَارَ الطَّبِيبُ الْمَدْرَسَةَ لِيَشْرَحَ لِلطُّلَّابِ أَهَمِّيَّةَ النَّظَافَةِ، وَتَحَدَّثَ عَنْ غَسْلِ الْيَدَيْنِ، وَشُرْبِ الْمَاءِ، وَالنَّوْمِ الْمُبَكِّرِ، وَمُمَارَسَةِ الرِّيَاضَةِ لِلْحِفَاظِ عَلَى صِحَّةٍ قَوِيَّةٍ وَجِسْمٍ نَشِيطٍ وَعَقْلٍ يَقِظٍ.",
    },
    {
        "plain": "في نهاية المؤتمر شكر المتحدث الحضور، وذكر أن التعاون بين الجامعات والشركات يفتح أبواب الابتكار، ويدعم الشباب في بناء مشاريع نافعة تخدم المجتمع بثقة ومسؤولية.",
        "gold": "فِي نِهَايَةِ الْمُؤْتَمَرِ شَكَرَ الْمُتَحَدِّثُ الْحُضُورَ، وَذَكَرَ أَنَّ التَّعَاوُنَ بَيْنَ الْجَامِعَاتِ وَالشَّرِكَاتِ يَفْتَحُ أَبْوَابَ الِابْتِكَارِ، وَيَدْعَمُ الشَّبَابَ فِي بِنَاءِ مَشَارِيعَ نَافِعَةٍ تَخْدُمُ الْمُجْتَمَعَ بِثِقَةٍ وَمَسْؤُولِيَّةٍ.",
    },
    {
        "plain": "حافظ الجار على هدوء الحي، وساعد الأسرة الجديدة في نقل الأثاث، ثم دعا الأطفال إلى اللعب بأمان قرب البيت مساء الجمعة بعد انتهاء واجباتهم المدرسية.",
        "gold": "حَافَظَ الْجَارُ عَلَى هُدُوءِ الْحَيِّ، وَسَاعَدَ الْأُسْرَةَ الْجَدِيدَةَ فِي نَقْلِ الْأَثَاثِ، ثُمَّ دَعَا الْأَطْفَالَ إِلَى اللَّعِبِ بِأَمَانٍ قُرْبَ الْبَيْتِ مَسَاءَ الْجُمُعَةِ بَعْدَ انْتِهَاءِ وَاجِبَاتِهِمُ الْمَدْرَسِيَّةِ.",
    },
]
DEMO_SENTENCES = [
    {
        "id": "student_library",
        "title": "Student And Library",
        "plain": "كتب الطالب الدرس ثم ذهب إلى المكتبة.",
        "gold": "كَتَبَ الطَّالِبُ الدَّرْسَ ثُمَّ ذَهَبَ إِلَى الْمَكْتَبَةِ.",
        "tag": "daily",
    },
    {
        "id": "teacher_question",
        "title": "Teacher Question",
        "plain": "سأل المعلم الطلاب عن معنى الكلمة.",
        "gold": "سَأَلَ الْمُعَلِّمُ الطُّلَّابَ عَنْ مَعْنَى الْكَلِمَةِ.",
        "tag": "classroom",
    },
    {
        "id": "morning_window",
        "title": "Morning Window",
        "plain": "في الصباح فتحت الأم النافذة ودخل الهواء.",
        "gold": "فِي الصَّبَاحِ فَتَحَتِ الْأُمُّ النَّافِذَةَ وَدَخَلَ الْهَوَاءُ.",
        "tag": "narrative",
    },
    {
        "id": "traveler_home",
        "title": "Traveler Home",
        "plain": "عاد المسافر إلى بيته بعد رحلة طويلة.",
        "gold": "عَادَ الْمُسَافِرُ إِلَى بَيْتِهِ بَعْدَ رِحْلَةٍ طَوِيلَةٍ.",
        "tag": "case endings",
    },
    {
        "id": "child_story",
        "title": "Child Story",
        "plain": "قرأ الطفل قصة قصيرة قبل النوم.",
        "gold": "قَرَأَ الطِّفْلُ قِصَّةً قَصِيرَةً قَبْلَ النَّوْمِ.",
        "tag": "tanween",
    },
    {
        "id": "honest_work",
        "title": "Honest Work",
        "plain": "إن العمل الصادق يبني الثقة بين الناس.",
        "gold": "إِنَّ الْعَمَلَ الصَّادِقَ يَبْنِي الثِّقَةَ بَيْنَ النَّاسِ.",
        "tag": "formal",
    },
    {
        "id": "word_count_25",
        "title": "25 Word Example",
        "plain": " ".join(segment["plain"] for segment in WORD_COUNT_DEMO_SEGMENTS[:1]),
        "gold": " ".join(segment["gold"] for segment in WORD_COUNT_DEMO_SEGMENTS[:1]),
        "tag": "25 words",
    },
    {
        "id": "word_count_50",
        "title": "50 Word Example",
        "plain": " ".join(segment["plain"] for segment in WORD_COUNT_DEMO_SEGMENTS[:2]),
        "gold": " ".join(segment["gold"] for segment in WORD_COUNT_DEMO_SEGMENTS[:2]),
        "tag": "50 words",
    },
    {
        "id": "word_count_100",
        "title": "100 Word Example",
        "plain": " ".join(segment["plain"] for segment in WORD_COUNT_DEMO_SEGMENTS[:4]),
        "gold": " ".join(segment["gold"] for segment in WORD_COUNT_DEMO_SEGMENTS[:4]),
        "tag": "100 words",
    },
    {
        "id": "word_count_250",
        "title": "250 Word Example",
        "plain": " ".join(segment["plain"] for segment in WORD_COUNT_DEMO_SEGMENTS[:10]),
        "gold": " ".join(segment["gold"] for segment in WORD_COUNT_DEMO_SEGMENTS[:10]),
        "tag": "250 words",
    },
]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def ensure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def stream_event(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    handler.wfile.write(body)
    handler.wfile.flush()


def html_response(handler: BaseHTTPRequestHandler) -> None:
    path = HTML_PATH if HTML_PATH.exists() else FALLBACK_HTML_PATH
    body = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def static_response(handler: BaseHTTPRequestHandler, request_path: str) -> bool:
    if not WEB_ROOT.exists():
        return False
    relative = request_path.lstrip("/")
    target = (WEB_ROOT / relative).resolve()
    try:
        target.relative_to(WEB_ROOT.resolve())
    except ValueError:
        return False
    if not target.is_file():
        return False
    body = target.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"))
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "public, max-age=31536000, immutable")
    handler.end_headers()
    handler.wfile.write(body)
    return True


def public_demo_sentences() -> list[dict]:
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "plain": item["plain"],
            "tag": item["tag"],
            "letters": len(extract_units(item["gold"])),
        }
        for item in DEMO_SENTENCES
    ]


def find_demo(demo_id: str) -> dict:
    for item in DEMO_SENTENCES:
        if item["id"] == demo_id:
            return item
    raise ValueError("Unknown demo sentence.")


def strip_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text)


def suppress_pause_punctuation_harakat(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        kept = "".join(mark for mark in match.group(2) if mark not in PAUSE_STRIP_MARKS)
        return match.group(1) + kept

    return PAUSE_PUNCTUATION_RE.sub(replace, text)


def restore_alif_forms(original: str, output: str) -> str:
    """Restore alif-form characters (hamza placement) from the original input.
    Diacritization models must not alter spelling — only harakat are their domain.
    """
    orig_bases = [ch for ch in original if ch not in TASHKEEL_CHARS and ch != "ـ"]
    result: list[str] = []
    idx = 0
    for ch in output:
        if ch in TASHKEEL_CHARS or ch == "ـ":
            result.append(ch)
        else:
            if idx < len(orig_bases):
                orig_ch = orig_bases[idx]
                result.append(orig_ch if (ch in ALIF_FORMS and orig_ch in ALIF_FORMS) else ch)
                idx += 1
            else:
                result.append(ch)
    return "".join(result)


def normalize_plain(text: str) -> str:
    return " ".join(strip_tashkeel(text).split()).strip()


def is_arabic_base(ch: str) -> bool:
    return bool(ARABIC_BASE_RE.fullmatch(ch))


def canonical_diacritics(diacs: str) -> str:
    if not diacs:
        return ""
    chars = set(diacs)
    if chars == {SUPERSCRIPT_ALIF}:
        return SUPERSCRIPT_ALIF
    vowel = ""
    for candidate in VOWELS:
        if candidate in chars:
            vowel = candidate
            break
    if SHADDA in chars and vowel:
        return vowel + SHADDA
    if SHADDA in chars:
        return SHADDA
    return vowel


def extract_units(text: str) -> list[tuple[str, str]]:
    units: list[list[str]] = []
    for ch in text:
        if is_arabic_base(ch):
            units.append([ch, ""])
        elif ch in TASHKEEL_CHARS and units:
            units[-1][1] += ch
    return [(base, canonical_diacritics(diacs)) for base, diacs in units]


def score_prediction(gold: str, predicted: str) -> dict:
    gold = suppress_pause_punctuation_harakat(gold)
    gold_units = extract_units(gold)
    pred_units = extract_units(predicted)
    total = len(gold_units)
    compared = min(len(gold_units), len(pred_units))
    base_errors = abs(len(gold_units) - len(pred_units))
    harakat_errors = 0
    correct = 0
    samples = []

    for index in range(compared):
        gold_base, gold_diacs = gold_units[index]
        pred_base, pred_diacs = pred_units[index]
        if gold_base != pred_base:
            base_errors += 1
            if len(samples) < 8:
                samples.append(
                    {
                        "index": index + 1,
                        "letter": gold_base,
                        "expected": gold_diacs,
                        "observed": pred_diacs,
                        "kind": "base",
                    }
                )
            continue
        if gold_diacs == pred_diacs:
            correct += 1
        else:
            harakat_errors += 1
            if len(samples) < 8:
                samples.append(
                    {
                        "index": index + 1,
                        "letter": gold_base,
                        "expected": gold_diacs or "none",
                        "observed": pred_diacs or "none",
                        "kind": "harakat",
                    }
                )

    errors = base_errors + harakat_errors
    if total:
        error_pct = min(100.0, (errors / total) * 100.0)
    else:
        error_pct = 0.0
    score_pct = max(0.0, 100.0 - error_pct)
    return {
        "score_pct": score_pct,
        "error_pct": error_pct,
        "correct": correct,
        "total": total,
        "errors": errors,
        "base_errors": base_errors,
        "harakat_errors": harakat_errors,
        "predicted_letters": len(pred_units),
        "perfect": errors == 0 and len(gold_units) == len(pred_units),
        "samples": samples,
    }


class RuntimeManager:
    def __init__(
        self,
        max_chars: int,
        enable_premium: bool,
        eo_batch_size: int,
        premium_batch_size: int,
        premium_cpu: bool,
    ) -> None:
        self.max_chars = max_chars
        self.enable_premium = enable_premium
        self.eo_batch_size = eo_batch_size
        self.premium_batch_size = premium_batch_size
        self.premium_cpu = premium_cpu
        self._eo_runners = {}
        self._premium_runner = None
        self._specialist = None
        self._specialist_vocab: dict | None = None
        self._lock = threading.Lock()

    @property
    def available_models(self) -> list[dict]:
        models = [
            {
                "id": "eo",
                "label": EO_MODEL_LABEL,
                "available": EO_CHECKPOINT.exists(),
                "loaded": self._eo_loaded,
                "description": "Polished Sukoon CATT EO runtime",
                "checkpoint": display_runtime_path(EO_CHECKPOINT),
            },
        ]
        if not self.enable_premium:
            return models
        models.extend([
            {
                "id": "premium",
                "label": "Shaddah",
                "available": self.enable_premium and self._premium_assets_exist(),
                "loaded": self._premium_loaded,
                "description": "Guarded Sukoon + AraModernBERT dual-stream runtime",
            },
            {
                "id": "tanween",
                "label": "Tanween",
                "available": self.enable_premium and self._tanween_assets_exist(),
                "loaded": self._tanween_loaded,
                "description": "Shaddah + final-letter specialist head",
            },
        ])
        return models

    @property
    def _eo_loaded(self) -> bool:
        return any(getattr(runner, "model", None) is not None for runner in self._eo_runners.values())

    @property
    def _premium_loaded(self) -> bool:
        return bool(self._premium_runner is not None and getattr(self._premium_runner, "head", None) is not None)

    @property
    def _tanween_loaded(self) -> bool:
        return self._premium_loaded and self._specialist is not None

    def _premium_assets_exist(self) -> bool:
        return HF_CHECKPOINT.exists() and DUAL_STREAM_HEAD.exists() and PREMIUM_CATT_CHECKPOINT.exists()

    def _tanween_assets_exist(self) -> bool:
        return self._premium_assets_exist() and SPECIALIST_CKPT.exists()

    @property
    def available_devices(self) -> list[dict]:
        gpu_ready = cuda_available()
        return [
            {
                "id": "cpu",
                "label": "CPU",
                "available": True,
                "loaded": "cpu" in self._eo_runners,
            },
            {
                "id": "gpu",
                "label": "GPU",
                "available": gpu_ready,
                "loaded": "cuda" in self._eo_runners,
                "description": "CUDA GPU" if gpu_ready else "CUDA is not available in this runtime",
            },
        ]

    def status(self) -> dict:
        return {
            "ok": True,
            "max_chars": self.max_chars,
            "models": self.available_models,
            "default_model": os.getenv("TASHKEEL_DEFAULT_MODEL", "eo"),
            "devices": self.available_devices,
            "default_device": self._safe_default_device(),
            "demos": public_demo_sentences(),
            "eo_checkpoint": display_runtime_path(EO_CHECKPOINT),
        }

    def preload(self, mode: str) -> None:
        mode = mode.lower()
        if mode in {"eo", "all"}:
            self._load_eo(os.getenv("TASHKEEL_DEVICE", "auto"))
        if mode in {"premium", "all"} and self.enable_premium:
            self._load_premium()
        if mode in {"tanween", "all"} and self.enable_premium:
            self._load_tanween()

    def diacritize(self, model_name: str, text: str, device_name: str = "auto") -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")
        model_name = (model_name or "eo").lower()
        if model_name not in {"eo", "premium", "tanween"}:
            raise ValueError("Unknown model.")
        if model_name in {"premium", "tanween"} and not self.enable_premium:
            raise ValueError("Premium is disabled in this container.")

        with self._lock:
            if model_name == "tanween":
                return self._run_tanween(text)
            if model_name == "premium":
                return self._run_premium(text)
            return self._run_eo(text, device_name)

    def diacritize_stream(self, model_name: str, text: str, device_name: str = "auto"):
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")
        model_name = (model_name or "eo").lower()
        if model_name != "eo":
            raise ValueError("Streaming is currently available for Sukoon only.")

        def events():
            with self._lock:
                yield from self._stream_eo(text, device_name)

        return events()

    def stats(self, demo_id: str, predicted: str, source_text: str = "") -> dict:
        predicted = predicted.strip()
        if not predicted:
            raise ValueError("Run a model before calculating stats.")
        demo = find_demo(demo_id)
        if source_text and normalize_plain(source_text) != normalize_plain(demo["plain"]):
            raise ValueError("Stats are available after loading an unchanged demo sentence.")
        stats = score_prediction(demo["gold"], predicted)
        return {
            **stats,
            "demo_id": demo["id"],
            "title": demo["title"],
            "plain": demo["plain"],
            "gold": demo["gold"],
            "method": "Perfect reference = 100%. Error % = mismatched Arabic-letter harakat / reference Arabic letters.",
        }

    def _resolve_device(self, device_name: str) -> str:
        device = (device_name or "auto").strip().lower()
        if device in {"auto", ""}:
            return "cuda" if cuda_available() else "cpu"
        if device == "gpu":
            device = "cuda"
        if device == "cuda" and not cuda_available():
            raise ValueError("GPU was requested, but CUDA is not available in this runtime.")
        if device not in {"cpu", "cuda"}:
            raise ValueError("Unknown device. Use cpu or gpu.")
        return device

    def _safe_default_device(self) -> str:
        try:
            device = self._resolve_device(os.getenv("TASHKEEL_DEVICE", "auto"))
            return "gpu" if device == "cuda" else device
        except ValueError:
            return "cpu"

    def _load_eo(self, device_name: str = "auto"):
        device = self._resolve_device(device_name)
        if device in self._eo_runners:
            return self._eo_runners[device]
        self._push_paths()
        from local_inference import LocalCATTInference

        runner = LocalCATTInference(
            checkpoint=str(EO_CHECKPOINT),
            batch_size=self.eo_batch_size,
            max_seq_len=1024,
            split_long_text=True,
            split_target_chars=int(os.getenv("TASHKEEL_EO_SPLIT_TARGET_CHARS", "320")),
            split_hard_chars=int(os.getenv("TASHKEEL_EO_SPLIT_HARD_CHARS", "520")),
            split_min_chars=int(os.getenv("TASHKEEL_EO_SPLIT_MIN_CHARS", "90")),
            split_soft_target_words=int(os.getenv("TASHKEEL_EO_SPLIT_SOFT_TARGET_WORDS", "0")),
            split_soft_min_words=int(os.getenv("TASHKEEL_EO_SPLIT_SOFT_MIN_WORDS", "5")),
            ml_boundary_split=env_bool("TASHKEEL_EO_ML_BOUNDARY_SPLIT", False),
            device=device,
        )
        runner.load()
        self._eo_runners[device] = runner
        return runner

    def _load_premium(self):
        if self._premium_runner is not None:
            return self._premium_runner
        if not self._premium_assets_exist():
            raise FileNotFoundError("Premium model assets are missing from runtime_models/premium.")
        self._push_paths()
        from premium_dual_stream_pipeline import PremiumDualStreamInference

        runner = PremiumDualStreamInference(
            hf_checkpoint_dir=str(HF_CHECKPOINT),
            dual_stream_head=str(DUAL_STREAM_HEAD),
            catt_checkpoint=str(PREMIUM_CATT_CHECKPOINT),
            catt_dir=str(CATT_DIR),
            batch_size=self.premium_batch_size,
            correction_scale=PREMIUM_CORRECTION_SCALE,
            critical_guard=True,
            cpu=self.premium_cpu,
        )
        runner.load()
        self._premium_runner = runner
        return runner

    def _load_tanween(self):
        self._load_premium()
        if self._specialist is not None:
            return
        if not SPECIALIST_CKPT.exists():
            raise FileNotFoundError("Specialist checkpoint missing from runtime_models/premium/runs/final_letter_specialist_v1/.")
        self._push_paths()
        import torch
        from final_letter_specialist import load_specialist
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, bv, dv, id2label = load_specialist(str(SPECIALIST_CKPT), device)
        self._specialist = (model, bv, dv, id2label, device)

    def _run_eo(self, text: str, device_name: str = "auto") -> dict:
        started = time.time()
        runner = self._load_eo(device_name)
        output, telemetry = runner.diacritize(text)
        output = restore_alif_forms(text, output)
        segment_info = telemetry["segment_info"]
        return {
            "model": "eo",
            "output": output,
            "elapsed": time.time() - started,
            "device": runner.device,
            "segments": segment_info.inferred_segments,
            "chunked": segment_info.chunked_segments,
            "warnings": segment_info.warnings,
            "split_methods": getattr(segment_info, "split_methods", {}),
            "max_segment_chars": getattr(segment_info, "max_segment_chars", 0),
        }

    def _stream_eo(self, text: str, device_name: str = "auto"):
        started = time.time()
        runner = self._load_eo(device_name)
        final_event = None
        for event in runner.iter_diacritize(text):
            if event.get("type") == "chunk":
                yield {"type": "chunk", "text": event["text"]}
                continue
            if event.get("type") == "done":
                final_output = restore_alif_forms(text, event.get("output", ""))
                final_event = {
                    "type": "done",
                    "model": "eo",
                    "output": final_output,
                    "elapsed": time.time() - started,
                    "device": runner.device,
                    "segments": event.get("segments", 0),
                    "chunked": event.get("chunked", 0),
                    "warnings": event.get("warnings", []),
                    "split_methods": event.get("split_methods", {}),
                    "max_segment_chars": event.get("max_segment_chars", 0),
                }
                continue
            yield {"type": "start", "model": "eo", **event}
        if final_event is not None:
            yield final_event

    def _run_premium(self, text: str) -> dict:
        started = time.time()
        runner = self._load_premium()
        result = runner.diacritize(text)
        info = result.segment_info
        return {
            "model": "premium",
            "output": result.output,
            "elapsed": time.time() - started,
            "device": str(runner.device),
            "segments": info.get("inferred_segments", 0),
            "chunked": info.get("chunked_segments", 0),
            "warnings": info.get("warnings", []),
            "split_methods": info.get("split_methods", {}),
            "max_segment_chars": info.get("max_segment_chars", 0),
            "critical_guard": info.get("critical_guard", {"enabled": True, "changes": []}),
        }

    def _run_tanween(self, text: str) -> dict:
        started = time.time()
        self._load_tanween()
        runner = self._premium_runner
        result = runner.diacritize(text)
        shaddah_elapsed = time.time() - started
        model, bv, dv, id2label, device = self._specialist
        from final_letter_specialist import apply_specialist
        corrected, n_edits = apply_specialist(result.output, model, bv, dv, id2label, device, SPECIALIST_THRESHOLD)
        info = result.segment_info
        return {
            "model": "tanween",
            "output": corrected,
            "elapsed": time.time() - started,
            "device": str(runner.device),
            "segments": info.get("inferred_segments", 0),
            "chunked": info.get("chunked_segments", 0),
            "warnings": info.get("warnings", []),
            "split_methods": info.get("split_methods", {}),
            "max_segment_chars": info.get("max_segment_chars", 0),
            "critical_guard": info.get("critical_guard", {"enabled": True, "changes": []}),
            "specialist_edits": n_edits,
            "specialist_threshold": SPECIALIST_THRESHOLD,
        }

    def _push_paths(self) -> None:
        for path in (str(HF_SCRIPTS), str(CATT_DIR)):
            if path not in sys.path:
                sys.path.insert(0, path)


def make_handler(manager: RuntimeManager):
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            if urlparse(self.path).path in {"/", "/healthz", "/api/status"}:
                self.send_response(200)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html_response(self)
                return
            if static_response(self, parsed.path):
                return
            if parsed.path == "/healthz":
                json_response(self, 200, {"ok": True})
                return
            if parsed.path == "/api/status":
                json_response(self, 200, manager.status())
                return
            if parsed.path == "/api/demos":
                json_response(self, 200, {"demos": public_demo_sentences()})
                return
            json_response(self, 404, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/diacritize", "/api/diacritize_stream", "/api/stats"}:
                json_response(self, 404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                max_body_bytes = int(os.getenv("TASHKEEL_MAX_BODY_BYTES", "0"))
                if max_body_bytes > 0 and length > max_body_bytes:
                    raise ValueError("Request body is too large.")
                raw_body = self.rfile.read(length)
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                if parsed.path == "/api/stats":
                    json_response(
                        self,
                        200,
                        manager.stats(
                            str(payload.get("demo_id", "")),
                            str(payload.get("predicted", "")),
                            str(payload.get("source_text", "")),
                        ),
                    )
                    return
                text = str(payload.get("text", ""))
                model_name = str(payload.get("model", os.getenv("TASHKEEL_DEFAULT_MODEL", "eo")))
                device_name = str(payload.get("device", os.getenv("TASHKEEL_DEVICE", "auto")))
                if parsed.path == "/api/diacritize_stream":
                    stream = manager.diacritize_stream(model_name, text, device_name)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    try:
                        for event in stream:
                            stream_event(self, event)
                    except Exception as exc:
                        traceback.print_exc()
                        stream_event(self, {"type": "error", "error": str(exc) or "Inference failed."})
                    return
                json_response(self, 200, manager.diacritize(model_name, text, device_name))
            except ValueError as exc:
                json_response(self, 400, {"error": str(exc)})
            except Exception:
                traceback.print_exc()
                json_response(self, 500, {"error": "Inference failed. Check the container logs for details."})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[demo] {self.address_string()} - {fmt % args}", flush=True)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local Tashkeel AI Docker demo.")
    parser.add_argument("--host", default=os.getenv("TASHKEEL_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TASHKEEL_PORT", "7860")))
    parser.add_argument("--public-url", default=os.getenv("TASHKEEL_PUBLIC_URL", "http://127.0.0.1:8765/"))
    parser.add_argument("--max-chars", type=int, default=int(os.getenv("TASHKEEL_MAX_CHARS", "0")))
    parser.add_argument("--preload", choices=["none", "eo", "premium", "tanween", "all"], default=os.getenv("TASHKEEL_PRELOAD", "none"))
    return parser.parse_args()


def main() -> None:
    ensure_utf8_console()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    args = parse_args()
    manager = RuntimeManager(
        max_chars=args.max_chars,
        enable_premium=env_bool("TASHKEEL_ENABLE_PREMIUM", True),
        eo_batch_size=int(os.getenv("TASHKEEL_EO_BATCH_SIZE", "32")),
        premium_batch_size=int(os.getenv("TASHKEEL_PREMIUM_BATCH_SIZE", "32")),
        premium_cpu=env_bool("TASHKEEL_PREMIUM_CPU", False),
    )
    if args.preload != "none":
        manager.preload(args.preload)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager))
    print("Tashkeel AI local container", flush=True)
    print(f"Open:     {args.public_url}", flush=True)
    print(f"Internal: http://{args.host}:{args.port}/", flush=True)
    print(f"Models:   {', '.join(m['label'] for m in manager.available_models if m['available'])}", flush=True)
    print("Press Ctrl+C or stop the container to exit.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
