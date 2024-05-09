FROM ajevnisek/snl-amir:v0
RUN pip install ipdb
COPY . ./
CMD ["/bin/bash", "-c", "source $EXPORT_SCRIPT && source $RUN_SCRIPT"]
