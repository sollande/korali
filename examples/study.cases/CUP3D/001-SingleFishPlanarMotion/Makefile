SWIMMER = run-vracer-swimmer eval-vracer-swimmer
KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUPFOLDER=../_deps/CUP-3D
CUBISMCFLAGS=$(shell cat ${CUPFOLDER}/makefiles/cup.cflags.txt) -I${CUPFOLDER}/source -I${CUPFOLDER}/Cubism/include -DCUBISM -Wno-shadow
CUBISMLIBS= ${CUPFOLDER}/makefiles/libcup.a $(shell cat ${CUPFOLDER}/makefiles/cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all
all: swimmer

swimmer: $(SWIMMER)

$(SWIMMER): %: %.o swimmerEnvironment.o ${CUPFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

swimmerEnvironment.o: _model/swimmerEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(RM) $(SWIMMER) *.txt *.o *.ti *.optrpt *.h5 *.dat *.xmf
