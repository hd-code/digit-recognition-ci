# build docker image (run from this dir)
docker build -t pandoc .

# start docker container with pandoc and latex (run from one dir above)
docker run --rm -it -v "${PWD}":/data pandoc

# build pdf file directly (run in docker container)
pandoc -o thesis.pdf --bibliography quellen.bib -H preamble.tex -B titlepage.tex *.md
pandoc -o projektabgabe.pdf --bibliography quellen.bib -H preamble.tex -B titlepage.tex *.md