import os
import re
from typing import Dict, List, Tuple

import cv2
from PIL import Image, ImageDraw
from pypdf import PdfMerger


def find_EOI(pipeline, matches: list, EOI: str) -> None:
    for obj in pipeline:
        group = obj["entity_group"]
        word = obj["word"]
        if group == EOI and len(word.strip("#")) > 3:
            temp = word.strip("#").split()
            [matches.append(w) for w in temp if len(w) > 2]


def draw_boxes_easyOCR(
    image: Image.Image, bounds: list, color: str = "yellow", width: int = 2
):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


def draw_boxes_pytesseract(
    image: Image.Image,
    data: Dict[str, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
) -> None:
    boxes = len(data["level"])
    for i in range(boxes):
        (x, y, w, h) = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        cv2.rectangle(image, (x, y), (x + w, y + h), color, width)


def draw_black_box_pytesseract(
    bbox: List[Tuple[int, int, int, int]],
    image: Image.Image,
    fill: str = "black",
    outline: str = "black",
) -> None:
    draw = ImageDraw.Draw(image)
    for box in bbox:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline)


def draw_black_box_easyOCR(
    bbox: List[Tuple[int, int, int, int]], image: Image.Image
) -> None:
    draw = ImageDraw.Draw(image)
    for box in bbox:
        p0, p1, p2, p3 = box
        draw.rectangle([*p0, *p2], fill="black", outline="black")


def find_coordinates_pytesseract(matches: List[str],
                                 data: dict,
                                 bbox: list) -> None:
    for obj in matches:
        for idx in range(len(data["text"])):
            i = data["text"][idx].strip()
            if i and obj in i:
                x, y, w, h = (
                    data["left"][idx],
                    data["top"][idx],
                    data["width"][idx],
                    data["height"][idx],
                )
                bbox.append((x, y, w, h))


def find_coordinates_easyOCR(pii_objects: List[str],
                             bounds: list,
                             bbox: list) -> None:
    for obj in pii_objects:
        for i in range(len(bounds)):
            if (obj.strip() in bounds[i][1].strip()) and (
                len(obj.strip()) > 3 and len(bounds[i][1].strip()) > 3
            ):
                bbox.append(bounds[i][0])


def find_emails(text: str, matches: list) -> None:
    match = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    matches += match


def find_numbers(text: str, matches: list) -> None:
    match = re.findall(r"[0-9]+", text)
    matches += match


def find_months(text: str, matches: list) -> None:
    match = re.findall(
        r"(?:January|February|March|April|May|June|July|August|\
        September|October|November|December)",
        text,
    )
    matches += match


def alter_metadata(file_name: str, output_name: str):
    with open(file_name, "rb") as file_in, open(output_name, "wb") as file_out:
        pdf_merger = PdfMerger()
        pdf_merger.append(file_in)
        pdf_merger.add_metadata({"/Author": "Unknown", "/Title": "Title"})
        pdf_merger.write(file_out)

    os.remove(file_name)
