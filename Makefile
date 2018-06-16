all : inplace

.PHONY : all

.FORCE :

inplace : .FORCE
	python setup.py build build_ext -f -i

radarkit : .FORCE
	python setup.py build build_ext -f

install :
	python setup.py install
	
clean :
	rm -rf build
	rm -f radarkit/*.so

