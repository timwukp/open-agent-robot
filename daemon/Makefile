# open-agent-robot macOS daemon
# deps: brew install ffmpeg libwebsockets   (miniaudio auto-fetched, vendored header)
BREW := $(shell brew --prefix)
CC = clang
CFLAGS = -O2 -Wall -Wextra -I$(BREW)/include
LDFLAGS = -L$(BREW)/lib \
  -lavdevice -lavformat -lavcodec -lswscale -lavutil -lwebsockets -lpthread \
  -framework CoreFoundation -framework CoreAudio -framework AudioToolbox \
  -framework AVFoundation -framework CoreMedia -framework CoreVideo \
  -framework VideoToolbox -framework AudioUnit

oar-daemon: main.c ring.h miniaudio.h
	$(CC) $(CFLAGS) main.c -o $@ $(LDFLAGS)

miniaudio.h:
	curl -fsSL https://raw.githubusercontent.com/mackron/miniaudio/master/miniaudio.h -o $@

clean:
	rm -f oar-daemon
.PHONY: clean
