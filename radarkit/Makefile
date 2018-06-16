all: rkstruct.so

rkstruct.so: setup.py rkstruct.c
	rm -f rkstruct*.so
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -f rkstruct*.so
