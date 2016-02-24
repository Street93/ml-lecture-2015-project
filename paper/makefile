
# copy all files from 'src' to 'build' and build there
.SILENT: build/doc.pdf
.PHONY: build/doc.pdf
build/doc.pdf: src/doc.tex src/bibliography.bib
	mkdir -p build 2>&1 > /dev/null; \
	cp src/* build; \
	cd build; \
	rubber --force --pdf doc > /dev/null;

clean:
	rm -r build
