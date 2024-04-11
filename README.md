# dib: [bmp/dib](https://en.wikipedia.org/wiki/BMP_file_format) format decoder

we decode your old windows file format.

## support table

|                        | enc | dec |
| ---------------------- | --- | --- |
| palette (2\|4\|8 bits) | ❎  | ✅  |
| 16\|24\|32 bpp         | ❎  | ✅  |
| run length (4\|8)      | ❎  | ✅  |
| bitfields              | 🟩  | ✅  |
| CMYK                   | ❎  | ❎  |
| PNG                    | ❎  | ❎  |
| JPEG                   | ❎  | ❎  |
