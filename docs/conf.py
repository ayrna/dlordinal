# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dlordinal"
copyright = "2023, Francisco Bérchez, Víctor Vargas"
author = "Francisco Bérchez, Víctor Vargas"
release = "2.6.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
bibtex_bibfiles = ["references.bib"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

# html_math_renderer = "imgmath"
# imgmath_image_format = "svg"
# imgmath_latex_preamble = "\\usepackage{fouriernc}"

autodoc_inherit_docstrings = True

# -- Options for LaTeX output -------------------------------------------------

latex_documents = [
    (
        "index",
        "dlordinal.tex",
        "dlordinal Documentation",
        "Francisco Bérchez, Víctor Vargas",
        "manual",
    ),
]

latex_engine = "xelatex"

latex_elements = {
    "preamble": r"""
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
""",
}
