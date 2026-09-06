"""Tests for language detection and test-file classification."""

from __future__ import annotations

import pytest

from backend.app.features.language import (
    LANGUAGE_LIST,
    detect_language,
    detect_language_index,
    is_prod_file,
    is_test_file,
)


class TestLanguageDetection:
    def test_python(self) -> None:
        assert detect_language("main.py") == "python"
        assert detect_language("src/utils.py") == "python"

    def test_python_variants(self) -> None:
        assert detect_language("script.pyw") == "python"

    def test_javascript(self) -> None:
        assert detect_language("app.js") == "javascript"
        assert detect_language("lib.mjs") == "javascript"
        assert detect_language("lib.cjs") == "javascript"

    def test_typescript(self) -> None:
        assert detect_language("app.ts") == "typescript"
        assert detect_language("app.tsx") == "typescript"

    def test_java(self) -> None:
        assert detect_language("Main.java") == "java"

    def test_go(self) -> None:
        assert detect_language("main.go") == "go"

    def test_rust(self) -> None:
        assert detect_language("lib.rs") == "rust"

    def test_c_cpp(self) -> None:
        assert detect_language("main.c") == "c"
        assert detect_language("main.h") == "c"
        assert detect_language("main.cpp") == "cpp"
        assert detect_language("main.cxx") == "cpp"

    def test_ruby(self) -> None:
        assert detect_language("app.rb") == "ruby"

    def test_shell(self) -> None:
        assert detect_language("build.sh") == "shell"

    def test_config_files(self) -> None:
        assert detect_language("config.yaml") == "yaml"
        assert detect_language("config.yml") == "yaml"
        assert detect_language("data.json") == "json"
        assert detect_language("pyproject.toml") == "toml"

    def test_web_files(self) -> None:
        assert detect_language("index.html") == "html"
        assert detect_language("style.css") == "css"

    def test_unknown_extension(self) -> None:
        assert detect_language("file.xyz") == "unknown"

    def test_no_extension(self) -> None:
        assert detect_language("Makefile") == "unknown"
        assert detect_language("Dockerfile") == "unknown"

    def test_case_insensitive(self) -> None:
        assert detect_language("APP.PY") == "python"
        assert detect_language("Main.JS") == "javascript"


class TestLanguageIndex:
    def test_python_index(self) -> None:
        idx = detect_language_index("main.py")
        assert 0.0 <= idx <= 1.0
        # Python is at index 0
        assert idx == 0.0

    def test_unknown_index(self) -> None:
        idx = detect_language_index("file.xyz")
        # Unknown is the last entry
        expected = (len(LANGUAGE_LIST) - 1) / (len(LANGUAGE_LIST) - 1)
        assert idx == expected

    def test_deterministic(self) -> None:
        assert detect_language_index("app.py") == detect_language_index("app.py")

    def test_all_languages_in_list(self) -> None:
        for lang in LANGUAGE_LIST:
            assert isinstance(lang, str)
        assert "python" in LANGUAGE_LIST
        assert "unknown" in LANGUAGE_LIST


class TestTestFileDetection:
    def test_test_directory(self) -> None:
        assert is_test_file("tests/test_main.py") is True
        assert is_test_file("test/unit/test_foo.py") is True

    def test_test_prefix(self) -> None:
        assert is_test_file("test_main.py") is True
        assert is_test_file("src/test_utils.py") is True

    def test_test_suffix(self) -> None:
        assert is_test_file("main_test.py") is True
        assert is_test_file("utils_test.js") is True

    def test_spec_suffix(self) -> None:
        assert is_test_file("app.spec.js") is True
        assert is_test_file("app.spec.ts") is True

    def test_spec_directory(self) -> None:
        assert is_test_file("spec/app_spec.rb") is True

    def test_prod_files_not_test(self) -> None:
        assert is_test_file("src/main.py") is False
        assert is_test_file("app.js") is False
        assert is_test_file("lib/utils.py") is False

    def test_empty_path(self) -> None:
        assert is_test_file("") is False

    def test_test_subdirectory(self) -> None:
        assert is_test_file("src/tests/helpers.py") is True
        assert is_test_file("src/test/unit.py") is True


class TestProdFileDetection:
    def test_prod_file(self) -> None:
        assert is_prod_file("src/main.py") is True

    def test_test_file_not_prod(self) -> None:
        assert is_prod_file("tests/test_main.py") is False

    def test_empty_path_is_prod(self) -> None:
        assert is_prod_file("") is True
