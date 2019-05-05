PYTHON=${PYTHON:-python}

all:
	echo "Compiling box_iou..."
	python setup.py build_ext --build-lib build
	@echo "from .box_iou import *" > build/__init__.py

clean:
	rm -rf build
