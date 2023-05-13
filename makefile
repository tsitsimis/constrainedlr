test:
	PYTHONPATH=. pytest tests

build:
	rm -r dist/
	rm -r *.egg-info
	python -m build


publish:
	python -m twine upload --repository pypi dist/*
