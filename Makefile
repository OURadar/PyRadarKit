all: rkstruct.so

rkstruct.so: setup.py rkstruct.c
	python3 setup.py build_ext

clean:
	rm -f rkstruct*.so
