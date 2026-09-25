import datetime
from typing import List, Optional
from app.session_manager import TranscriptEvent


def _format_timestamp_vtt(seconds_float: float) -> str:
    """Format seconds into WebVTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds_float // 3600)
    minutes = int((seconds_float % 3600) // 60)
    seconds = int(seconds_float % 60)
    millis = int(round((seconds_float - int(seconds_float)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _format_timestamp_srt(seconds_float: float) -> str:
    """Format seconds into SubRip SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds_float // 3600)
    minutes = int((seconds_float % 3600) // 60)
    seconds = int(seconds_float % 60)
    millis = int(round((seconds_float - int(seconds_float)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def export_vtt(events: List[TranscriptEvent], lang: str = "es", base_time: Optional[float] = None) -> str:
    """
    Generates clean WebVTT subtitles from committed final events.
    Excludes interims and empty transcripts.
    """
    final_events = [e for e in events if e.type == "final" and e.original.strip()]
    if not final_events:
        return "WEBVTT\n\nNOTE Empty transcript\n"

    first_ts = base_time if base_time is not None else final_events[0].timestamp
    lines = ["WEBVTT\n"]

    for i, event in enumerate(final_events, start=1):
        rel_start = max(0.0, event.timestamp - first_ts)
        # Default subtitle duration: 3.5 seconds or estimate based on word count
        duration = max(2.0, min(6.0, len(event.original.split()) * 0.4))
        rel_end = rel_start + duration

        start_str = _format_timestamp_vtt(rel_start)
        end_str = _format_timestamp_vtt(rel_end)

        # Get text for requested language (or original)
        if lang == "original" or lang == event.source_language:
            text = event.original.strip()
        else:
            text = event.translations.get(lang, event.original).strip()

        speaker_prefix = f"<v {event.speaker}>" if event.speaker else ""
        speaker_suffix = "</v>" if event.speaker else ""

        lines.append(f"{i}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{speaker_prefix}{text}{speaker_suffix}\n")

    return "\n".join(lines)


def export_srt(events: List[TranscriptEvent], lang: str = "es", base_time: Optional[float] = None) -> str:
    """
    Generates standard SubRip SRT subtitles from committed final events.
    """
    final_events = [e for e in events if e.type == "final" and e.original.strip()]
    if not final_events:
        return "1\n00:00:00,000 --> 00:00:02,000\n[Empty transcript]\n"

    first_ts = base_time if base_time is not None else final_events[0].timestamp
    blocks = []

    for i, event in enumerate(final_events, start=1):
        rel_start = max(0.0, event.timestamp - first_ts)
        duration = max(2.0, min(6.0, len(event.original.split()) * 0.4))
        rel_end = rel_start + duration

        start_str = _format_timestamp_srt(rel_start)
        end_str = _format_timestamp_srt(rel_end)

        if lang == "original" or lang == event.source_language:
            text = event.original.strip()
        else:
            text = event.translations.get(lang, event.original).strip()

        speaker_header = f"[{event.speaker}] " if event.speaker else ""
        blocks.append(f"{i}\n{start_str} --> {end_str}\n{speaker_header}{text}\n")

    return "\n".join(blocks)


def export_txt(events: List[TranscriptEvent], lang: str = "es") -> str:
    """
    Generates plain-text conference transcript with timestamps.
    """
    final_events = [e for e in events if e.type == "final" and e.original.strip()]
    if not final_events:
        return "[Empty transcript]\n"

    lines = []
    for event in final_events:
        time_str = datetime.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        if lang == "original" or lang == event.source_language:
            text = event.original.strip()
        else:
            text = event.translations.get(lang, event.original).strip()
        speaker = f"{event.speaker}: " if event.speaker else ""
        lines.append(f"[{time_str}] {speaker}{text}")

    return "\n".join(lines) + "\n"
