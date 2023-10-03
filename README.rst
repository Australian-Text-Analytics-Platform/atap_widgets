atap_widgets
==============

|PyPI| |Status| |Python Version|

|Read the Docs| |Tests| |License|

|pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/atap_widgets.svg
   :target: https://pypi.org/project/atap_widgets/
   :alt: PyPI
.. |Status| image:: https://img.shields.io/pypi/status/atap_widgets.svg
   :target: https://pypi.org/project/atap_widgets/
   :alt: Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/atap_widgets
   :target: https://pypi.org/project/atap_widgets
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/atap_widgets
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/atap_widgets/latest.svg?label=Read%20the%20Docs
   :target: https://atap_widgets.readthedocs.io/
   :alt: Read the documentation at https://atap_widgets.readthedocs.io/
.. |Tests| image:: https://github.com/Australian-Text-Analytics-Platform/atap_widgets/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/Australian-Text-Analytics-Platform/atap_widgets/actions?workflow=Tests
   :alt: Tests
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

This package is part of the atap_ project.

Features
--------

* Conversation recurrence plotting
* Concordance search and export


Requirements
------------

* Python 3.8+
* spacy
* pandas
* Interactive widgets are designed for use in Jupyter Lab (3+)


Installation
------------

You can install *atap_widgets* via pip_ from PyPI_:

.. code:: console

   $ pip install atap_widgets

**NOTE**: on M1 Macs with the new Apple Silicon chip, you may need a Rust compiler installed. Rust can be installed with a single command via https://rustup.rs/ . You may also need `cmake`: install via Homebrew with `brew install cmake`.


Standalone tools
------------

Concordaner
Use the Concordancer as a standalone tool by clicking the following Binder link

.. image:: https://binderhub.atap-binder.cloud.edu.au/badge_logo.svg
 :target: https://binderhub.atap-binder.cloud.edu.au/v2/gh/Australian-Text-Analytics-Platform/atap_widgets/concordance_standalone?labpath=concordance_standalone.ipynb

Development
------------

This project uses poetry_ for dependency management and
packaging. Please see the poetry_ docs for details.
Dependencies are specified in ``pyproject.toml``.

This repository is configured to automatically publish
new releases to PyPI if you merge a pull request with
an updated version number. That means to release
a new version with new features/fixes, you should:

* Create a branch
* Perform your work on that branch
* Update the package version, e.g. with ``poetry version patch`` or ``poetry version minor``
* Commit the changes to ``pyproject.toml``
* Create a pull request for the branch on GitHub
* Merge the branch into ``main`` when you're ready to release


Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `MIT license`_,
*atap_widgets* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


Credits
-------

This project was generated from `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.

.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MIT license: https://opensource.org/licenses/MIT
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _file an issue: https://github.com/Australian-Text-Analytics-Platform/atap_widgets/issues
.. _pip: https://pip.pypa.io/
.. _poetry: https://python-poetry.org/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://atap_widgets.readthedocs.io/en/latest/usage.html
.. _atap: https://www.atap.edu.au/
