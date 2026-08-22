import numpy as np
import matplotlib.pyplot as plt

t0 = 6.095910535477454e-03

# Common slant-range positions of 20 fine DCE points
tau = np.array([
    6.106567335918665e-03,
    6.125067534994453e-03,
    6.143589056264030e-03,
    6.162110573895627e-03,
    6.180632098803183e-03,
    6.199132334258759e-03,
    6.217653837338442e-03,
    6.236175340418124e-03,
    6.254696828945892e-03,
    6.273218332025574e-03,
    6.291718531101363e-03,
    6.310240034181045e-03,
    6.328761639124134e-03,
    6.347283142203817e-03,
    6.365804645283499e-03,
    6.384304844359288e-03,
    6.402826347438970e-03,
    6.421347821414822e-03,
    6.439869324494505e-03,
    6.458390943989509e-03,
])


DCE = [
    {
        "name": "DCE1",
        "azimuthTime": "2025-12-26T21:43:59.100490",

        "geometry": np.array([
            -1.049767e+00,
            -9.459936e+01,
             3.584473e+04
        ]),

        "data": np.array([
             4.152910e+01,
             1.012491e+05,
            -4.252661e+08
        ]),

        "fine": np.array([
            45.70864868164062,
            44.04507827758789,
            43.03631973266602,
            44.01594161987305,
            47.99888992309570,
            43.02584075927734,
            57.71871948242188,
            43.69139099121094,
            50.88753128051758,
            42.66687011718750,
            38.58018875122070,
            40.42238998413086,
            40.04460144042969,
            42.24584960937500,
            47.02595138549805,
            36.08031082153320,
            32.20111846923828,
            32.15531921386719,
            27.85684013366699,
            15.96245002746582,
        ])
    },

    {
        "name": "DCE2",
        "azimuthTime": "2025-12-26T21:44:14.095877",

        "geometry": np.array([
             3.093618e+00,
            -2.768187e+02,
             3.257385e+04
        ]),

        "data": np.array([
            1.141131e+01,
            1.275731e+04,
            6.579813e+07
        ]),

        "fine": np.array([
            18.81233024597168,
            10.75786972045898,
            12.51480007171631,
            11.24419021606445,
            11.24717044830322,
            11.75018978118896,
             7.781082153320312,
             5.143774986267090,
            16.67377090454102,
            16.29187011718750,
            16.71276092529297,
            23.39012908935547,
            23.86281013488770,
            21.72710037231445,
            18.75226974487305,
            20.16904067993164,
            25.48768997192383,
            21.80849075317383,
            22.97163009643555,
            19.53133964538574,
        ])
    },

    {
        "name": "DCE3",
        "azimuthTime": "2025-12-26T21:44:25.484967",

        "geometry": np.array([
             1.370909e+00,
             3.247131e+01,
            -2.398021e+04
        ]),

        "data": np.array([
             3.263240e+01,
            -2.579159e+03,
            -1.314992e+08
        ]),

        "fine": np.array([
            45.53495025634766,
            34.35984039306641,
            29.68753051757812,
            30.57007980346680,
            31.76118087768555,
            31.07452964782715,
            30.56966018676758,
            32.79946136474609,
            29.16184997558594,
            26.26977920532227,
            25.97484016418457,
            28.13009071350098,
            25.23070907592773,
            24.25044059753418,
            20.96414947509766,
            19.80022048950195,
            19.18678092956543,
            17.33567047119141,
            15.62742042541504,
            16.16124916076660,
        ])
    }
]


def eval_poly(coeff, tau):
    dt = tau - t0
    return coeff[0] + coeff[1]*dt + coeff[2]*dt**2


for dce in DCE:

    x_dense = np.linspace(tau.min(), tau.max(), 1000)

    y_data = eval_poly(dce["data"], x_dense)
    y_geom = eval_poly(dce["geometry"], x_dense)

    # Polynomial evaluated exactly at fitting points
    fit_data = eval_poly(dce["data"], tau)
    fit_geom = eval_poly(dce["geometry"], tau)

    residual_data = dce["fine"] - fit_data
    residual_geom = dce["fine"] - fit_geom

    SSE_data = np.sum(residual_data**2)
    SSE_geom = np.sum(residual_geom**2)

    RMSE_data = np.sqrt(np.mean(residual_data**2))
    RMSE_geom = np.sqrt(np.mean(residual_geom**2))

    print()
    print(dce["name"], dce["azimuthTime"])
    print(f"SSE data     = {SSE_data:.6f} Hz^2")
    print(f"SSE geometry = {SSE_geom:.6f} Hz^2")
    print(f"RMSE data    = {RMSE_data:.6f} Hz")
    print(f"RMSE geometry= {RMSE_geom:.6f} Hz")

    plt.figure(figsize=(9, 5))

    plt.scatter(
        (tau - t0) * 1e6,
        dce["fine"],
        label="Fine DCE points"
    )

    plt.plot(
        (x_dense - t0) * 1e6,
        y_data,
        label="Data DC polynomial"
    )

    plt.plot(
        (x_dense - t0) * 1e6,
        y_geom,
        label="Geometry DC polynomial"
    )

    plt.xlabel(r"$\tau-t_0$ [$\mu$s]")
    plt.ylabel("Doppler centroid [Hz]")
    plt.title(
        f'{dce["name"]} - {dce["azimuthTime"]}\n'
        f'Data RMSE={RMSE_data:.3f} Hz, '
        f'Geometry RMSE={RMSE_geom:.3f} Hz'
    )

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()