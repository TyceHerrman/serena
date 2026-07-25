import pytest

from serena.memories.frontmatter import FrontmatterParser, parse_frontmatter, render_frontmatter


def test_parse_without_frontmatter_returns_content_unchanged() -> None:
    content = "# Memory\n\nBody\n"

    result = FrontmatterParser.parse(content)

    assert result.frontmatter == {}
    assert result.body == content


def test_parse_frontmatter_preserves_body_whitespace() -> None:
    content = '---\nsummary: "Short description"\nurl: https://example.com:443/docs\n---\n\n# Memory\n\nBody\n'

    result = FrontmatterParser.parse(content)

    assert result.frontmatter == {
        "summary": "Short description",
        "url": "https://example.com:443/docs",
    }
    assert result.body == "\n# Memory\n\nBody\n"


def test_parse_empty_frontmatter() -> None:
    result = FrontmatterParser.parse("---\n---\nBody\n")

    assert result.frontmatter == {}
    assert result.body == "Body\n"


@pytest.mark.parametrize(
    "content",
    [
        "---\nsummary: missing closing delimiter\n",
        "---\nnot a scalar field\n---\nBody\n",
        "---\n: missing key\n---\nBody\n",
        " ---\nsummary: indented opening delimiter\n---\nBody\n",
    ],
)
def test_malformed_frontmatter_is_plain_content(content: str) -> None:
    result = FrontmatterParser.parse(content)

    assert result.frontmatter == {}
    assert result.body == content


def test_render_round_trip_is_stable() -> None:
    frontmatter = {
        "summary": "Short description",
        "custom": "value:with:colons",
    }
    body = "\n# Memory\n\nBody with trailing whitespace.\n\n"

    rendered = FrontmatterParser.render(frontmatter, body)
    parsed = FrontmatterParser.parse(rendered)

    assert parsed.frontmatter == frontmatter
    assert parsed.body == body
    assert FrontmatterParser.render(parsed.frontmatter, parsed.body) == rendered


def test_functional_wrappers_match_parser() -> None:
    content = "---\nsummary: Notes\n---\nBody"

    frontmatter, body = parse_frontmatter(content)

    assert frontmatter == {"summary": "Notes"}
    assert body == "Body"
    assert render_frontmatter(frontmatter, body) == content
