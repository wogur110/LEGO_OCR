"""Generate vector A4 LEGO marker boards using OpenCV and Ghostscript.

Run with the LEGO_OCR virtual environment. Poppler supplies PDF previews.
Neither the recognizer nor its calibration is changed by this generator.
"""

import argparse
import html
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 297.0, 210.0
SIDE = 36.0
POINTS_PER_MM = 72.0 / 25.4
PLACEMENTS = [("TL", 0, 15, 15), ("TR", 1, 246, 15),
              ("BL", 2, 15, 159), ("BR", 3, 246, 159)]


class Page:
    def __init__(self):
        self.ps = ["%!PS-Adobe-3.0", "%%Pages: 1", "%%Page: 1 1",
                   f"<< /PageSize [{WIDTH * POINTS_PER_MM:.8f} "
                   f"{HEIGHT * POINTS_PER_MM:.8f}] >> setpagedevice"]
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{WIDTH}mm" height="{HEIGHT}mm" '
                    f'viewBox="0 0 {WIDTH} {HEIGHT}">',
                    '<rect width="297" height="210" fill="white"/>']

    def rect(self, x, y, width, height, gray=0):
        p = POINTS_PER_MM
        self.ps.append(f"{gray} setgray {x*p:.8f} {(HEIGHT-y-height)*p:.8f} "
                       f"{width*p:.8f} {height*p:.8f} rectfill")
        self.svg.append(f'<rect x="{x:.8f}" y="{y:.8f}" '
                        f'width="{width:.8f}" height="{height:.8f}" '
                        f'fill="rgb({gray*100}%,{gray*100}%,{gray*100}%)"/>')

    def text(self, x, y, value, size=8, gray=0.28, bold=False):
        p = POINTS_PER_MM
        escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        font = "Helvetica-Bold" if bold else "Helvetica"
        self.ps.append(f"{gray} setgray /{font} findfont {size} scalefont setfont "
                       f"{x*p:.8f} {(HEIGHT-y)*p:.8f} moveto "
                       f"({escaped}) dup stringwidth pop -2 div 0 rmoveto show")
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" '
                        f'font-family="Helvetica,Arial,sans-serif" '
                        f'font-size="{size/p:.8f}" '
                        f'font-weight="{"bold" if bold else "normal"}" '
                        f'fill="rgb({gray*100}%,{gray*100}%,{gray*100}%)">'
                        f'{html.escape(value)}</text>')

    def save(self, path):
        with tempfile.TemporaryDirectory(prefix="lego-board-render-") as directory:
            source = Path(directory) / "board.ps"
            source.write_text("\n".join([*self.ps, "showpage", "%%EOF"]) + "\n")
            subprocess.run([
                "gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE",
                "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
                "-dEmbedAllFonts=true", f"-sOutputFile={path}", str(source)
            ], check=True)
        path.with_suffix(".svg").write_text("\n".join([*self.svg, "</svg>"]) + "\n")


def marker_matrix(kind, corner, identifier):
    if kind == "ArUco":
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        return cv2.aruco.generateImageMarker(dictionary, identifier, 6), identifier, 1
    parameters = cv2.QRCodeEncoder_Params()
    parameters.version = 1
    parameters.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_M
    payload = "LEGO-A4:" + corner
    encoded = cv2.QRCodeEncoder_create(parameters).encode(payload)
    # OpenCV includes padding; retain the 21x21 version-1 symbol itself.
    rows, columns = np.where(encoded == 0)
    core = encoded[rows.min():rows.max()+1, columns.min():columns.max()+1]
    assert core.shape == (21, 21)
    return core, payload, 4


def make_board(kind, output):
    page = Page()
    page.text(WIDTH / 2, 19, "LEGO WORKSPACE", size=13, gray=0.15, bold=True)
    subtitle = ("ARUCO 4X4_100  /  CORNER IDs 0-3" if kind == "ArUco"
                else "QR BOARD  /  UNIQUE CORNER CODES")
    page.text(WIDTH / 2, 25, subtitle, size=7)
    markers = []
    for corner, identifier, x, y in PLACEMENTS:
        matrix, value, quiet_modules = marker_matrix(kind, corner, identifier)
        module = SIDE / len(matrix)
        for row in range(len(matrix)):
            # Draw contiguous black runs, preserving exact vector module sizes.
            column = 0
            while column < len(matrix):
                if matrix[row, column] != 0:
                    column += 1
                    continue
                end = column + 1
                while end < len(matrix) and matrix[row, end] == 0:
                    end += 1
                page.rect(x + column*module, y + row*module,
                          (end-column)*module, module)
                column = end
        label = f"{corner} / ID {identifier}" if kind == "ArUco" else corner
        page.text(x + SIDE/2, 63 if corner[0] == "T" else 147, label, size=7)
        markers.append({
            "corner": corner, "id" if kind == "ArUco" else "payload": value,
            "symbol_side_mm": SIDE, "modules_per_side": len(matrix),
            "quiet_zone_min_mm": quiet_modules * module,
            "corners_mm_tl_tr_br_bl": [[x, y], [x+SIDE, y],
                                       [x+SIDE, y+SIDE], [x, y+SIDE]],
        })

    # The central placement region remains entirely white, with no printed grid.
    scale_x, scale_y = (WIDTH-50)/2, 188
    page.rect(scale_x, scale_y, 50, 0.25)
    page.rect(scale_x, scale_y-1.4, 0.25, 3)
    page.rect(scale_x+50-0.25, scale_y-1.4, 0.25, 3)
    page.text(WIDTH/2, 195, "50 mm - check with a ruler", size=7)
    page.text(WIDTH/2, 203,
              "A4 LANDSCAPE  /  PRINT AT 100% (ACTUAL SIZE)  /  NO FIT TO PAGE",
              size=7)

    stem = output / f"lego_board_A4_{kind}"
    page.save(stem.with_suffix(".pdf"))
    for dpi, suffix in [(300, "reference"), (120, "preview")]:
        subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", str(dpi),
                        str(stem.with_suffix(".pdf")), str(stem)+"_"+suffix],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    metadata = {
        "version": 1, "board_id": f"lego-a4-landscape-{kind.lower()}-v1",
        "marker_family": "DICT_4X4_100" if kind == "ArUco" else "QR_CODE",
        "qr_version": 1 if kind == "QR" else None,
        "qr_error_correction": "M" if kind == "QR" else None,
        "page_mm": [WIDTH, HEIGHT], "orientation": "landscape",
        "coordinates": "millimeters; origin at page top-left; +x right; +y down",
        "print_scale": 1.0, "scale_bar_mm": 50,
        "clear_center_xyxy_mm": [65, 43, 232, 167],
        "clear_center_size_mm": [167, 124],
        "markers": markers, "reference_raster_dpi": 300,
        "reference_raster": stem.name + "_reference.png",
        "notes": [
            "Marker coordinates describe the printed paper plane, not brick tops.",
            "This is board geometry, not a calibrated BrailleReader slot layout.",
            "Existing slot layouts and empty references must be recalibrated for this board.",
            "The QR board needs a QR detector; BoardAligner currently uses ArUco.",
        ],
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ["QR", "ArUco"]:
        make_board(kind, args.output_dir)
    print(f"Generated both A4 boards in {args.output_dir}")


if __name__ == "__main__":
    main()
