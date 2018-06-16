SUBDIRS := radarkit

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: all $(SUBDIRS)
