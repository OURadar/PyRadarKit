all : inplace

.PHONY : all

.FORCE :

inplace : .FORCE
	find . -name '*.o' -name '*.so' -delete
	python setup.py build build_ext -f -i

radarkit : .FORCE
	python setup.py build build_ext -f

install :
	python setup.py install --record files.txt
	
clean :
	rm -rf build dist *.egg-info .ipynb_checkpoints
	rm -f radarkit/*.so

