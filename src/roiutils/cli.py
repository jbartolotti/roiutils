"""Command line interface for roiutils."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib

from .atlas_io import load_labels_tsv
from .interface import build_roi_mask, load_atlas, resolve_roi_selection


def _parse_selection(raw: list[str]) -> list[int | str]:
    parsed: list[int | str] = []
    for token in raw:
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            parsed.append(int(token))
        else:
            parsed.append(token)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROI atlas subset tools")
    parser.add_argument("--atlas", required=True, help="Path to atlas NIfTI image")
    parser.add_argument("--labels", required=True, help="Path to labels TSV with id and label columns")
    parser.add_argument(
        "--select",
        nargs="+",
        required=True,
        help="ROI selectors (IDs or exact labels)",
    )
    parser.add_argument("--output", required=True, help="Output path for mask NIfTI")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when a selector does not match an ROI",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON path to save resolved ROI metadata",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    labels_by_id = load_labels_tsv(args.labels)
    atlas = load_atlas(args.atlas, labels_by_id)
    selection = resolve_roi_selection(atlas, _parse_selection(args.select), strict=args.strict)
    mask = build_roi_mask(atlas, selection)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(mask, str(output_path))

    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "atlas": str(args.atlas),
            "selected_ids": list(selection.ids),
            "selected_labels": selection.labels_by_id,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
