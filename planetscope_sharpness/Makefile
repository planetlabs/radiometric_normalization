
CFLAGS+=-O3 -march=native -g
CXXFLAGS+=-O3 -march=native -g -std=c++11
CXXFLAGS+=-Wall -Wextra

# for tvdeconv_20120607/
CFLAGS+=-DNUM_SINGLE -DTVREG_DECONV=1

LIBS+=-ljpeg -lpng -ltiff -lfftw3f

all: estimate-kernel deconv

OBJS=iio.o downscale.o upsa.o

estimate-kernel: src/estimate_kernel.cpp src/image.cpp ${OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS}

deconv: src/deconv.cpp tvdeconv_20120607/tvreg.o tvdeconv_20120607/basic.o src/image.cpp ${OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS} -DNUM_SINGLE -Itvdeconv_20120607/

%: %.cpp src/image.cpp ${OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@ ${LIBS}

%.o: imscript/%.c
	${CC} ${CFLAGS} -c $^ -o $@ -DHIDE_ALL_MAINS -DOMIT_MAIN

clean:
	-rm ${OBJS} estimate-kernel deconv
	-rm tvdeconv_20120607/tvreg.o tvdeconv_20120607/basic.o

