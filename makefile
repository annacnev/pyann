LIB_DIR = pyann/annlib/lib
ANNLIB_DIR = pyann/annlib

default: pyann

pyann: setup.py $(ANNLIB_DIR)/annlib.pyx $(LIB_DIR)/libNN.a
	python3 setup.py build_ext --inplace && rm -f $(ANNLIB_DIR)/annlib.cpp && rm -Rf build

$(LIB_DIR)/libNN.a:
	make -C $(LIB_DIR) libNN.a

clean:
	rm *.so