"""DEM provider tùy chọn: WGS84 geographic, độ cao ellipsoid tính bằng mét."""
import numpy as np


class RasterTerrain:
    def __init__(self, path, *, height_datum, height_unit):
        if height_datum!='ellipsoid' or height_unit!='m':
            raise ValueError('DEM cần datum ellipsoid và height_unit=m được khai báo tường minh.')
        try:
            import rasterio
        except ImportError as error:
            raise RuntimeError('DEM cần optional dependency: pip install -e ".[dem]"') from error
        self.dataset=rasterio.open(path)
        try:
            if self.dataset.crs!=rasterio.crs.CRS.from_epsg(4326) or self.dataset.count!=1:
                raise ValueError('DEM bản đầu cần một band và CRS EPSG:4326.')
            tags={k.lower():v.lower() for k,v in self.dataset.tags().items()}
            if any(word in tags.get('vertical_datum','') for word in ('orthometric','egm','geoid')):
                raise ValueError('DEM vertical datum mâu thuẫn với khai báo ellipsoid.')
            if self.dataset.units[0] not in (None,'m','metre','meter'):
                raise ValueError('DEM band unit không phải mét.')
        except Exception:
            self.dataset.close()
            raise

    def __enter__(self):return self
    def __exit__(self,*args):self.dataset.close()

    def __call__(self, latitude_deg, longitude_deg):
        latitude=float(latitude_deg);longitude=float(longitude_deg)
        bounds=self.dataset.bounds
        if (not np.isfinite([latitude,longitude]).all() or not bounds.left<=longitude<bounds.right
                or not bounds.bottom<latitude<=bounds.top):
            raise ValueError('Terrain query ngoài DEM coverage.')
        value=next(self.dataset.sample([(longitude,latitude)],indexes=1,masked=True))[0]
        if np.ma.is_masked(value) or not np.isfinite(value):
            raise ValueError('Terrain query chạm DEM nodata/NaN.')
        return float(value)
