all: rkstruct.so

rkstruct.so: setup.py rkstruct.c
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -f rkstruct*.so
