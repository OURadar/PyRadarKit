all : inplace

.PHONY : all

.FORCE :

inplace : .FORCE
	rm -rf build
	find . -name '*.o' -name '*.so' -delete
	python3 setup.py build build_ext -f -i

radarkit : .FORCE
	python3 setup.py build build_ext -f

install :
	python3 setup.py install --record files.txt
	
clean :
	rm -rf build dist *.egg-info .ipynb_checkpoints
	rm -f radarkit/*.so
