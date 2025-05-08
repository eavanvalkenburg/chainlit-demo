from pathlib import Path
from typing import Annotated
from pydantic import BaseModel
from semantic_kernel.data import (
    vectorstoremodel,
    VectorStoreRecordKeyField,
    VectorStoreRecordDataField,
    VectorStoreRecordVectorField,
)
from semantic_kernel.connectors.memory import ChromaCollection
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

import re


@vectorstoremodel
class DocsEntries(BaseModel):
    title: Annotated[str, VectorStoreRecordKeyField()]
    description: Annotated[str, VectorStoreRecordDataField(is_full_text_indexed=True)]
    author: Annotated[str, VectorStoreRecordDataField(is_indexed=True)]
    content: Annotated[str, VectorStoreRecordDataField(is_full_text_indexed=True)]
    filename: Annotated[str, VectorStoreRecordDataField(is_indexed=True)]
    embedding: Annotated[
        list[float] | str | None, VectorStoreRecordVectorField(dimensions=1536)
    ] = None

    def model_post_init(self, context):
        if self.embedding is None:
            self.embedding = f"{self.title} - {self.description} - {self.content}"


def remove_zone_pivot_tags(text):
    # Remove all ::: zone pivot="..." and ::: zone-end lines
    text = re.sub(
        r'^::: zone pivot="programming-language-python"\\s*$',
        "",
        text,
        flags=re.MULTILINE,
    )
    # Remove lines like ::: zone-end
    text = re.sub(r"^::: zone-end\\s*$", "", text, flags=re.MULTILINE)
    return text


# First, remove all zone-pivot blocks except for python
# We'll keep python blocks, but clean them up
# Remove all non-python zone-pivot blocks
def remove_non_python_zone_pivots(text):
    # Remove all zone pivots except python
    pattern = re.compile(
        r"""::: zone pivot=\\?\"(?!programming-language-python)[^\"]*\\?\"[\s\S]*?::: zone-end""",
        re.MULTILINE,
    )
    return pattern.sub("", text)


def read_data(file_path) -> DocsEntries:
    """
    Reads data from a file and returns it as a DocsEntries object.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        DocsEntries: A DocsEntries object containing the data from the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Find YAML frontmatter between the first two '---' lines
    frontmatter = {}
    in_frontmatter = False
    frontmatter_lines = []
    frontmatter_end_idx = 0
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            else:
                frontmatter_end_idx = idx
                break
        if in_frontmatter:
            frontmatter_lines.append(line)

    # Parse frontmatter lines for title, description, author
    for line in frontmatter_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip().lower()] = value.strip()

    # The rest is content
    content = remove_zone_pivot_tags(
        remove_non_python_zone_pivots("".join(lines[frontmatter_end_idx + 1 :]))
    )

    return DocsEntries(
        title=frontmatter.get("title", ""),
        description=frontmatter.get("description", ""),
        author=frontmatter.get("author", ""),
        content=content.strip(),
        filename=str(file_path),
    )


def read_folder(folder_path):
    """
    Reads all files in a folder and returns their contents as a list of lines.

    Args:
        folder_path (str): The path to the folder to read.

    Returns:
        list: A list of lines from all files in the folder.
    """

    all_entries = []
    for filename in folder_path.iterdir():
        entry = read_data(filename)
        all_entries.append(entry)
    return all_entries


async def main():
    folder_path = Path.cwd() / "data" / "markdowns"
    output_path = Path.cwd() / "data" / "output.jsonl"

    all_entries = read_folder(folder_path)

    async with ChromaCollection(
        collection_name="docs",
        data_model_type=DocsEntries,
        embedding_generator=OpenAITextEmbedding(),
        persist_directory=str(Path.cwd() / "data" / "chroma"),
    ) as chroma:
        await chroma.create_collection()
        await chroma.upsert(all_entries)

    # Write the entries to a JSONL file (one compact JSON object per line)
    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        for entry in all_entries:
            jsonl_file.write(entry.model_dump_json(exclude_none=True, indent=4) + "\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
