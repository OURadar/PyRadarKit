all: rkstruct.so

rkstruct.so: setup.py rkstruct.c
	python3 setup.py build_ext --inplace

clean:
	rm -f rkstruct*.so
