from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile

from sentinel1_processing.geotiff import export_slc_with_gcps


ANNOTATION = """<product>
  <imageAnnotation><imageInformation>
    <numberOfLines>2</numberOfLines><numberOfSamples>3</numberOfSamples>
  </imageInformation></imageAnnotation>
  <geolocationGrid><geolocationGridPointList count="2">
    <geolocationGridPoint><line>0</line><pixel>0</pixel><latitude>10</latitude>
      <longitude>20</longitude><height>1</height></geolocationGridPoint>
    <geolocationGridPoint><line>1</line><pixel>2</pixel><latitude>11</latitude>
      <longitude>22</longitude><height>2</height></geolocationGridPoint>
  </geolocationGridPointList></geolocationGrid>
</product>"""


class GeoTiffTest(unittest.TestCase):
    def test_export_slc_with_l1_gcps(self):
        slc = np.arange(6, dtype=np.float32).reshape(2, 3) + 1j
        with TemporaryDirectory() as directory:
            annotation = Path(directory) / "annotation.xml"
            annotation.write_text(ANNOTATION)
            output = export_slc_with_gcps(
                slc, annotation, Path(directory) / "slc.tiff"
            )

            with tifffile.TiffFile(output) as product:
                page = product.pages[0]
                np.testing.assert_array_equal(
                    page.asarray(),
                    np.rint(slc.real).astype(np.int16)
                    + 1j * np.rint(slc.imag).astype(np.int16),
                )
                self.assertEqual(page.tags["BitsPerSample"].value, 32)
                self.assertEqual(page.tags["SamplesPerPixel"].value, 1)
                self.assertEqual(page.tags["SampleFormat"].value, 5)
                self.assertFalse(product.is_bigtiff)
                self.assertEqual(page.tags[33922].value[:6], (0, 0, 0, 20, 10, 1))
                self.assertIn(4326, page.tags[34735].value)


if __name__ == "__main__":
    unittest.main()
