all: trace

.PHONY: trace
trace: 
	xelatex beamer_file
	gs -sDEVICE=ps2write -dNOCACHE -sOutputFile=out.ps -q -dbatch -dNOPAUSE beamer_file.pdf -c quit
