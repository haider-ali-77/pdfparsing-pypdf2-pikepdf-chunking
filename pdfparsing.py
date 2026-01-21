from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Union, Tuple, Dict, Any

from enum import Enum
from grobid.client import GrobidClient
import json
import os
import pikepdf
import PyPDF2
from PyPDF2 import PdfFileReader
import requests
import shutil
import tempfile
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET

from app.core.data.doc_style_tagging import (
    tags_info,
    bold_italic_tag_info,
    string_matching,
)
sentence_length = 7

XML_NAMESPACES = {
    "main": "http://www.w3.org/1999/xhtml",
    "grobid": "http://www.tei-c.org/ns/1.0",
}

def page_to_dict(txt):
    """Gets text in the form of list and convert it into the dictionary.
    Args:
        txt (List): List containing text of a page at each instance.
    Returns:
        dictionary (dic): Dictionary containing text of pages.
    """
    pages = txt
    return {
        "pages": [block_to_dict(page) for page in pages],
    }


def block_to_dict(txt):
    """Gets text in the form of list and convert it into the dictionary.
    Args:
        txt (List): List containing text of a block at each instance.
    Returns:
        dictionary (dic): Dictionary containing text of blocks.
    """
    blocks = txt.split("\#\#\#")  # noqa
    return {
        "blocks": [paragraph_to_dict(block) for block in blocks],
    }


def paragraph_to_dict(txt):
    """Gets text in the form of list and convert it into the dictionary.
    Args:
        txt (List): List containing text of a paragraph at each instance.
    Returns:
        dictionary (dic): Dictionary containing text of paragraphs.
    """
    paragraphs = txt.split("\#\#")  # noqa
    return {
        "paragraphs": [line_to_dict(paragraph) for paragraph in paragraphs],
    }


def line_to_dict(txt):
    """Gets text in the form of list and convert it into the dictionary.
    Args:
        txt (List): List containing text of a line at each instance.
    Returns:
        dictionary (dic): Dictionary containing text of lines.
    """
    lines = txt.split("\#")  # noqa
    return {
        "lines": [word_to_dict(line) for line in lines],
    }


def word_to_dict(txt):
    """Gets a word in the form of string and converts it into
    the dictionary containing the word.
    Args:
        txt (str): Containing a word.
    Returns:
        dictionary (dic): Dictionary containing text of a word.
    """
    all_words = txt
    return {
        "words": [all_words],
    }


def _pdf_list_to_texts(input_list, output_dir_json, output_dir_tagged_json):

    for i, input_path in enumerate(input_list):
        name = input_path.split("/")[-1]
        name = name.split(".")[:-1]
        name = ".".join(name)
        output_dir = Path(f"{output_dir_json}/{name}.json")
        output_dir_tagged = Path(f"{output_dir_tagged_json}/{name}.json")

        if PdfFileReader(open(input_path, "rb")).isEncrypted:
            pdf = pikepdf.open(input_path, allow_overwriting_input=True)
            pdf.save(input_path)

        try:
            _extract_text_from_pdf(
                pdf_path=input_path,
                output_path_json=output_dir,
                output_path_tagged_json=output_dir_tagged,
            )
        except Exception as e:
            return "error parsing " + str(e)

    return "successful parsing"


def _extract_text_from_pdf(
    pdf_path: Path, output_path_json: Path, output_path_tagged_json: Path
):

    pdf_converter = PDFConverter()
    try:
        pdf_text = pdf_converter.pdf_to_dict(
            input_file=pdf_path, output_path_tagged_json=output_path_tagged_json
        )
    except (PyPDF2.utils.PdfReadError, IndexError) as e:
        print(f"Extraction failed for file: {pdf_path}, {e}")
        return

    output_path_json.parents[0].mkdir(exist_ok=True, parents=True)
    with open(output_path_json, "w") as f:
        json.dump(pdf_text, f)


XML_NAMESPACES = {
    "main": "http://www.w3.org/1999/xhtml",
    "grobid": "http://www.tei-c.org/ns/1.0",
}


class GROBIDHeader:
    title_path = [
        "grobid:teiHeader",
        "grobid:fileDesc",
        "grobid:titleStmt",
        "grobid:title",
    ]
    authors_main_path = [
        "grobid:teiHeader",
        "grobid:fileDesc",
        "grobid:sourceDesc",
        "grobid:biblStruct",
        "grobid:analytic",
    ]
    forename_path = ["grobid:persName", "grobid:forename"]
    surname_path = ["grobid:persName", "grobid:surname"]

    def __init__(self, title: str, authors: List[str]):
        self.title = title
        self.authors = authors

    @classmethod
    def from_grobid_xml(cls, xml):
        try:
            tree = ET.fromstring(xml)
        except Exception as e:  # noqa
            return cls("", [])

        title = cls.get_title_from_tree(tree)
        authors = cls.get_authors_from_tree(tree)
        return cls(title=title, authors=authors)

    @classmethod
    def get_title_from_tree(cls, tree: ET):
        title = tree
        for element in cls.title_path:
            title = title.find(element, namespaces=XML_NAMESPACES)
        return title.text

    @classmethod
    def get_authors_from_tree(cls, tree):
        next = tree
        for element in cls.authors_main_path:
            next = next.find(element, namespaces=XML_NAMESPACES)
        authors_blocks = next.findall("grobid:author", namespaces=XML_NAMESPACES)
        authors = []
        for author_blocks in authors_blocks:
            try:
                forename, surname = author_blocks, author_blocks
                for element in cls.forename_path:
                    forename = forename.find(element, namespaces=XML_NAMESPACES)
                for element in cls.surname_path:
                    surname = surname.find(element, namespaces=XML_NAMESPACES)
                authors.append(f"{forename.text} {surname.text}")
            except AttributeError:
                continue
        return authors

    def __repr__(self):
        return f"{self.__class__.__name__}(title={self.title}, authors=({self.authors})"


class PDFConverter:
    def __init__(
        self,
        grobid_client: Optional[GrobidClient] = None,
        image_density: int = 300,
        image_depth: int = 8,
        image_format: str = "png",
    ):
        self.grobid_client = grobid_client
        self.image_density = image_density
        self.image_depth = image_depth
        self.image_format = image_format

    def pdf_to_image(
        self,
        input_file: Path,
        output_dir: Path,
    ):
        num_pages = self.get_num_pages(input_file)

        all_status = []
        # iterate through pages
        for i in range(num_pages):
            # Convert PDF to image format
            cmd = (
                f"convert -density {self.image_density} -depth {self.image_depth} "
                f"'{input_file}[{i}]' -background white '{output_dir}/{i}.{self.image_format}'"
            )
            print(cmd)

            status = os.system(cmd)
            print(status)
            all_status.append(status)
        return all_status

    def image_to_hocr(
        self, input_dir: Path, output_dir: Optional[Path] = None, num_pages: int = 1
    ):
        output_dir = input_dir if output_dir is None else output_dir

        all_status = []
        for i in range(num_pages):
            cmd = f"tesseract '{input_dir}/{i}.{self.image_format}' '{output_dir}/{i}' hocr"
            print(cmd)
            status = os.system(cmd)
            print(status)
            all_status.append(status)
        return all_status

    def image_to_text_pdf(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        num_pages: int = 1,
    ):
        output_dir = input_dir if output_dir is None else Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        all_status = []
        for i in range(num_pages):
            cmd = f"tesseract '{input_dir}/{i}.{self.image_format}' '{output_dir}/{i}' pdf"
            print(cmd)
            status = os.system(cmd)
            all_status.append(status)
        return all_status

    def hocr_to_pdf_documents(
        self,
        input_dir: Path,
        num_pages: int,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
    ):
        documents = []
        for page in range(num_pages):
            with open(f"{input_dir}/{page}.hocr") as f:
                page_str = f.read()
            document = PDFDocument.from_hocr(page_str, title=title, authors=authors)
            documents.append(document)
        return documents

    def _pdf_document_to_pages_dict(self, pdf_documents):
        pages = [pdf_document.to_dict()["pages"][0] for pdf_document in pdf_documents]
        for i, page in enumerate(pages):
            page["number"] = i  # Add page number to each page
        return pages

    def hocr_to_dict(
        self,
        file_path: Path,
        output_path_tagged_json: Path,
        input_dir: Path,
        num_pages: int,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        resize_to: Optional[Tuple[int, int]] = None,
    ):
        pdf_documents = self.hocr_to_pdf_documents(
            input_dir=input_dir, num_pages=num_pages, title=title, authors=authors
        )

        tagged_pages = []

        for i in range(len(pdf_documents)):

            text_ocr = pdf_documents[i].to_text_json()

            headings_font, footers_font, paragraph_font, list_text_font = tags_info(
                file_path, i
            )

            if not (
                (len(headings_font) == 0)
                and (len(footers_font) == 0)
                and (len(paragraph_font) == 0)
                and (len(list_text_font) == 0)
            ):

                text_list = list_text_font[0]
                font_list = list_text_font[1]

                main_list = bold_italic_tag_info(file_path, i, "<b>")
                text_list_bold = main_list[0]
                bold_list = main_list[1]

                main_list = bold_italic_tag_info(file_path, i, "<i>")
                text_list_italic = main_list[0]
                italic_list = main_list[1]

                text_ocr = string_matching(
                    text_list, font_list, text_ocr, thresh=sentence_length
                )
                text_ocr = string_matching(
                    text_list_bold, bold_list, text_ocr, thresh=sentence_length
                )
                text_ocr = string_matching(
                    text_list_italic, italic_list, text_ocr, thresh=sentence_length
                )

                tagged_pages.append(text_ocr)

        tagged_json = page_to_dict(tagged_pages)

        output_path_tagged_json.parents[0].mkdir(exist_ok=True, parents=True)

        out_file = open(f"{output_path_tagged_json}", "w")

        json.dump(tagged_json, out_file, indent=6)
        out_file.close()

        pdf_documents = [
            pdf_document.resize_to(resize_to) for pdf_document in pdf_documents
        ]  # Resize to PDF size
        pages = self._pdf_document_to_pages_dict(pdf_documents)
        return {
            "pages": pages,
            "title": title,
            "authors": authors,
        }

    def text_pdf_to_metadata_info(self, input_dir: Path) -> GROBIDHeader:
        if self.grobid_client is None:
            return GROBIDHeader("", [])
        try:
            response, status = self.grobid_client.serve(
                "processHeaderDocument", f"{input_dir}/0.pdf"
            )
            xml_content = response.content
            return GROBIDHeader.from_grobid_xml(xml_content)
        except requests.exceptions.ConnectionError:
            return GROBIDHeader("", [])

    def get_pdf_size(self, input_file: Path) -> Tuple[int, int]:
        with open(input_file, "rb") as f:
            input_pdf = PdfFileReader(f)
            media_box = input_pdf.getPage(0).mediaBox

        min_pt = media_box.lowerLeft
        max_pt = media_box.upperRight

        pdf_width = max_pt[0] - min_pt[0]
        pdf_height = max_pt[1] - min_pt[1]
        return pdf_width, pdf_height

    def get_num_pages(self, input_file: Path):
        with open(str(input_file), "rb") as fp:
            num_pages = PdfFileReader(fp).getNumPages()
        return num_pages

    def pdf_to_dict(self, input_file: Path, output_path_tagged_json: Path):

        num_pages = self.get_num_pages(input_file)

        tmp_dir = tempfile.mkdtemp()
        size = self.get_pdf_size(input_file)
        self.pdf_to_image(input_file, tmp_dir)
        self.image_to_hocr(
            input_dir=tmp_dir,
            num_pages=num_pages,
            output_dir=tmp_dir,
        )
        self.image_to_text_pdf(input_dir=tmp_dir, num_pages=1, output_dir=tmp_dir)
        metadata = self.text_pdf_to_metadata_info(input_dir=tmp_dir)
        return self.hocr_to_dict(
            input_dir=tmp_dir,
            num_pages=num_pages,
            title=metadata.title,
            authors=metadata.authors,
            resize_to=size,
            file_path=input_file,
            output_path_tagged_json=output_path_tagged_json,
        )

        # ... do stuff with dirpath
        shutil.rmtree(tmp_dir)


class PDFStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"


class BoundingBox(BaseModel):
    x_small: int
    y_small: int
    x_high: int
    y_high: int
    width: int
    height: int

    def __init__(self, **data: Any):
        for i, x in data.items():
            if not isinstance(x, int):
                raise TypeError(
                    f"Parameter {i} not of type integer: {x}. "
                    f"Use method from_coordinates to initialize from floats."
                )
        super().__init__(**data)
        if self.width != self.x_high - self.x_small:
            raise ValueError(f"{self.width=} != {self.x_high=} - {self.x_small=}")
        if self.height != self.y_high - self.y_small:
            raise ValueError(f"{self.height=} != {self.y_high=} - {self.y_small=}")

    @staticmethod
    def from_coordinates(x_small: int, y_small: int, x_high: int, y_high: int):
        x_small, y_small, x_high, y_high = (
            int(x_small),
            int(y_small),
            int(x_high),
            int(y_high),
        )
        width = x_high - x_small
        height = y_high - y_small
        return BoundingBox(
            x_small=x_small,
            y_small=y_small,
            x_high=x_high,
            y_high=y_high,
            width=width,
            height=height,
        )

    @staticmethod
    def from_element(element: Element):
        title_label = dict(element.items())["title"]
        properties = title_label.split("; ")
        for property in properties:
            if property.startswith("bbox"):
                string_bbox = property.replace("bbox ", "")
                break
        bounding_box_coordinates = tuple([int(x) for x in string_bbox.split()])
        return BoundingBox.from_coordinates(*bounding_box_coordinates)

    def resize_by_factor(
        self,
        resize_factor: Optional[Union[float, Tuple[float, float]]] = None,
        resize_x: Optional[float] = None,
        resize_y: Optional[float] = None,
    ):
        if resize_factor is None and (resize_x is None or resize_y is None):
            raise ValueError(
                "If resize_factor is None, resize_x and resize_y must not be None"
            )
        elif resize_factor is not None and (
            resize_x is not None and resize_y is not None
        ):
            raise ValueError(
                "If resize_factor is not None, resize_x and resize_y must be None"
            )
        if resize_factor is not None:
            if isinstance(resize_factor, tuple) and len(resize_factor) != 2:
                raise ValueError(f"resize_factor must have length 2: {resize_factor=}")
            elif isinstance(resize_factor, tuple):
                resize_x, resize_y = resize_factor
            else:
                resize_x, resize_y = resize_factor, resize_factor
        x_small = int(self.x_small * resize_x)
        x_high = int(self.x_high * resize_x)
        y_small = int(self.y_small * resize_y)
        y_high = int(self.y_high * resize_y)
        return BoundingBox.from_coordinates(
            x_small=x_small, x_high=x_high, y_small=y_small, y_high=y_high
        )

    def overlaps_with(self, other: "BoundingBox") -> bool:
        x_small = max([self.x_small, other.x_small])
        y_small = max([self.y_small, other.y_small])
        x_high = min([self.x_high, other.x_high])
        y_high = min([self.y_high, other.y_high])

        if x_small < x_high and y_small < y_high:
            return True
        else:
            return False


class Word:
    def __init__(
        self, word: str, bounding_box: BoundingBox, page: Optional[int] = None
    ):
        self.word = word
        self.bounding_box = bounding_box
        self.size = bounding_box.width, bounding_box.height
        self.page = page

    def resize_by_factor(self, resize_factor: Tuple[float, float]):
        new_bounding_box = self.bounding_box.resize_by_factor(resize_factor)
        return Word(word=self.word, bounding_box=new_bounding_box)

    def to_text(self):
        return self.word

    def to_text_json(self):
        return self.word

    def to_dict(self):
        return {"word": self.word, "bounding_box": dict(self.bounding_box)}

    @classmethod
    def from_hocr_word(cls, word: Element):
        bounding_box = BoundingBox.from_element(word)
        return Word(word.text, bounding_box)

    @classmethod
    def from_dict(cls, word):
        return cls(
            word=word["word"],
            bounding_box=BoundingBox(**word["bounding_box"]),
        )

    def __str__(self):
        return self.to_text()


class Line:
    def __init__(
        self,
        words: List[Word],
        bounding_box: Optional[BoundingBox] = None,
        page: Optional[int] = None,
    ):
        self.words = words
        self.bounding_box = bounding_box
        self.size = bounding_box.width, bounding_box.height
        self._page = page

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, page):
        self._page = page
        for word in self.words:
            word.page = page

    def resize_by_factor(self, resize_factor: Tuple[float, float]):
        new_bounding_box = self.bounding_box.resize_by_factor(resize_factor)
        new_words = [word.resize_by_factor(resize_factor) for word in self.words]
        return Line(words=new_words, bounding_box=new_bounding_box)

    def to_text(self):
        string = [word.to_text() for word in self.words]
        return " ".join(string)

    def to_text_json(self):
        """Join words of sentence
        Returns:
            String (str): String made by joining words.
        """
        string = [word.to_text_json() for word in self.words]
        return " ".join(string)

    def to_dict(self):
        """Join words of sentence
        Returns:
            Dictionary (dic): Dictionary containing words and their bounding box .
        """
        all_words = ""
        for word in self.words:

            all_words = all_words + " " + word.to_text()
        return {
            "words": [all_words],
            "bounding_box": dict(self.bounding_box),
        }

    @classmethod
    def from_hocr_line(cls, line: Element):
        word_strings = line.findall("main:span", namespaces=XML_NAMESPACES)
        bounding_box = BoundingBox.from_element(line)
        return Line([Word.from_hocr_word(line) for line in word_strings], bounding_box)

    @classmethod
    def from_dict(cls, line):
        return cls(
            words=[Word.from_dict(word) for word in line["words"]],
            bounding_box=BoundingBox(**line["bounding_box"]),
        )

    def __str__(self):
        return self.to_text()


class Paragraph:
    def __init__(
        self,
        lines: List[Line],
        bounding_box: Optional[BoundingBox] = None,
        page: Optional[int] = None,
    ):
        self.lines = lines
        self.bounding_box = bounding_box
        self.size = bounding_box.width, bounding_box.height
        self._page = page

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, page):
        self._page = page
        for line in self.lines:
            line.page = page

    def resize_by_factor(self, resize_factor: Tuple[float, float]):
        new_bounding_box = self.bounding_box.resize_by_factor(resize_factor)
        new_lines = [line.resize_by_factor(resize_factor) for line in self.lines]
        return Paragraph(lines=new_lines, bounding_box=new_bounding_box)

    def to_text(self):
        string = [line.to_text() for line in self.lines]
        return "\n".join(string)

    def to_text_json(self):
        """Join lines of a paragraph
        Returns:
            String (str): Paragraph.
        """
        string = [line.to_text_json() for line in self.lines]
        return "\#".join(string)  # noqa

    def to_dict(self):
        return {
            "lines": [line.to_dict() for line in self.lines],
            "bounding_box": dict(self.bounding_box),
        }

    @classmethod
    def from_hocr_paragraph(cls, paragraph: Element):
        line_strings = paragraph.findall("main:span", namespaces=XML_NAMESPACES)
        bounding_box = BoundingBox.from_element(paragraph)
        return Paragraph(
            [Line.from_hocr_line(line) for line in line_strings], bounding_box
        )

    @classmethod
    def from_dict(cls, paragraph):
        return cls(
            lines=[Line.from_dict(line) for line in paragraph["lines"]],
            bounding_box=BoundingBox(**paragraph["bounding_box"]),
        )

    def __str__(self):
        return self.to_text()


class Block:
    def __init__(
        self,
        paragraphs: List[Paragraph],
        bounding_box: Optional[BoundingBox] = None,
        page: Optional[int] = None,
    ):
        self.paragraphs = paragraphs
        self.bounding_box = bounding_box
        self.size = bounding_box.width, bounding_box.height
        self._page = page

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, page):
        self._page = page
        for paragraph in self.paragraphs:
            paragraph.page = page

    def resize_by_factor(self, resize_factor: Tuple[float, float]):
        new_bounding_box = self.bounding_box.resize_by_factor(resize_factor)
        new_paragraphs = [
            paragraph.resize_by_factor(resize_factor) for paragraph in self.paragraphs
        ]
        return Block(paragraphs=new_paragraphs, bounding_box=new_bounding_box)

    def to_text(self):
        string = [paragraph.to_text() for paragraph in self.paragraphs]
        return "\n\n".join(string)

    def to_text_json(self):
        string = [paragraph.to_text_json() for paragraph in self.paragraphs]
        return "\#\#".join(string)  # noqa

    def to_dict(self):
        return {
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "bounding_box": dict(self.bounding_box),
        }

    @classmethod
    def from_hocr_block(cls, block: Element):
        paragraph_strings = block.findall("main:p", namespaces=XML_NAMESPACES)
        bounding_box = BoundingBox.from_element(block)
        return Block(
            [
                Paragraph.from_hocr_paragraph(paragraph)
                for paragraph in paragraph_strings
            ],
            bounding_box,
        )

    @classmethod
    def from_dict(cls, block):
        return cls(
            paragraphs=[
                Paragraph.from_dict(paragraph) for paragraph in block["paragraphs"]
            ],
            bounding_box=BoundingBox(**block["bounding_box"]),
        )

    def __str__(self):
        return self.to_text()


class Page:
    def __init__(
        self,
        blocks: List[Block],
        bounding_box: Optional[BoundingBox] = None,
        page: Optional[int] = None,
    ):
        self.blocks = blocks
        self.bounding_box = bounding_box
        self.size = bounding_box.width, bounding_box.height
        self._page = page

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self, page):
        self._page = page
        for block in self.blocks:
            block.page = page

    def resize_by_factor(self, resize_factor: Tuple[float, float]):
        new_bounding_box = self.bounding_box.resize_by_factor(resize_factor)
        new_blocks = [block.resize_by_factor(resize_factor) for block in self.blocks]
        return Page(blocks=new_blocks, bounding_box=new_bounding_box)

    def to_text(self):
        string = [block.to_text() for block in self.blocks]
        return "\n\n".join(string)

    def to_text_json(self):
        """Join blocks of a page
        Returns:
            String (str): Page text.
        """
        string = [block.to_text_json() for block in self.blocks]
        return "\#\#\#".join(string)  # noqa

    def to_dict(self):

        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "bounding_box": dict(self.bounding_box),
        }

    @classmethod
    def from_hocr_page(cls, page: Element):

        blocks_strings = page.findall("main:div", namespaces=XML_NAMESPACES)
        bounding_box = BoundingBox.from_element(page)
        return Page(
            [Block.from_hocr_block(block) for block in blocks_strings], bounding_box
        )

    @classmethod
    def from_dict(cls, page):
        return cls(
            blocks=[Block.from_dict(block) for block in page["blocks"]],
            bounding_box=BoundingBox(**page["bounding_box"]),
        )

    def __str__(self):
        return self.to_text()


class PDFDocument:
    iterate_values = {"pages", "blocks", "paragraphs", "lines", "words"}

    def __init__(
        self,
        pages: List[Page],
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        id: str = None,
        iterate_on: str = "paragraphs",
    ):
        self.pages = self._initialize_pages(pages)
        self.title = title
        self.authors = authors
        self.id = id

        if iterate_on not in self.iterate_values:
            raise ValueError(f"iterate_on must be among {self.iterate_values}")
        self.iterate_on = iterate_on
        self._elements_as_list = list(self)

    def _initialize_pages(self, pages: List[Page]):
        # pages = deepcopy(pages)
        for i, page in enumerate(pages):
            page.page = i
        return pages

    def resize_to(self, new_size: Tuple[int, int]):
        pages_size = self.pages[0].size
        resize_factor = new_size[0] / pages_size[0], new_size[1] / pages_size[1]
        new_pages = [page.resize_by_factor(resize_factor) for page in self.pages]
        return PDFDocument(pages=new_pages, title=self.title, authors=self.authors)

    def to_text(self):
        string = [page.to_text() for page in self.pages]
        return "".join(string)

    def to_text_json(self):
        string = [page.to_text_json() for page in self.pages]
        return "".join(string)

    def to_dict(self):

        return {
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_hocr(
        cls,
        hocr_string: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
    ):
        try:
            tree = ET.fromstring(hocr_string)

        except Exception as e:
            print(e)
            return PDFDocument(pages=[], title=title, authors=authors)
        body = tree.find("main:body", namespaces=XML_NAMESPACES)

        pages_strings = body.findall("main:div", namespaces=XML_NAMESPACES)

        return PDFDocument(
            [Page.from_hocr_page(page) for page in pages_strings],
            title=title,
            authors=authors,
        )

    @classmethod
    def from_dict(cls, document: Dict):
        return cls(pages=[Page.from_dict(page) for page in document["pages"]])

    def __iter__(self):
        pages = (page for page in self.pages)
        blocks = (block for page in pages for block in page.blocks)
        paragraphs = (paragraph for block in blocks for paragraph in block.paragraphs)
        lines = (line for paragraph in paragraphs for line in paragraph.lines)
        words = (word for line in lines for word in line.words)
        if self.iterate_on == "pages":
            return pages
        elif self.iterate_on == "blocks":
            return blocks
        elif self.iterate_on == "paragraphs":
            return paragraphs
        elif self.iterate_on == "lines":
            return lines
        elif self.iterate_on == "words":
            return words
        else:
            raise ValueError(f"iterate on must be among {self.iterate_values}")

    def __getitem__(self, item):
        return self._elements_as_list[item]

    def __len__(self):
        lengths = [0, 0, 0, 0, 0]
        for page in self.pages:
            lengths[0] += 1
            for block in page.blocks:
                lengths[1] += 1
                for paragraph in block.paragraphs:
                    lengths[2] += 1
                    for line in paragraph.lines:
                        lengths[3] += 1
                        for word in line.words:
                            lengths[4] += 1

        if self.iterate_on == "pages":
            return lengths[0]
        elif self.iterate_on == "blocks":
            return lengths[1]
        elif self.iterate_on == "paragraphs":
            return lengths[2]
        elif self.iterate_on == "lines":
            return lengths[3]
        elif self.iterate_on == "words":
            return lengths[4]
        else:
            raise ValueError(f"iterate on must be among {self.iterate_values}")

    def __str__(self):
        return self.to_text()
