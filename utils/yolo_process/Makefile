PYTHON=${PYTHON:-python}

all:
	echo "Compiling yolo..."
	python setup.py build_ext --build-lib build
	echo "from .yolo_to_box import *" > build/__init__.py

clean:
	rm -rf build
