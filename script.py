from sirilpy import SirilInterface, LogColor, NoImageError
from sirilpy.utility import download_with_progress, ensure_installed, SuppressedStderr

ensure_installed("astropy", "numpy")

import tempfile
from astropy.io import fits
import numpy as np

siril = SirilInterface()
siril.connect()


# Modified from the GraXpert-AI script
def save_fits(data, path, original_header=None):
    if data.dtype not in (np.float32, np.uint16):
        data = data.astype(np.float32)
    if original_header is None:
        header = fits.Header()
    else:
        try:
            with SuppressedStderr():
                header = fits.Header.fromstring(original_header, sep='\n')
        except:
            header = fits.Header()
    fits.writeto(path, data, header, overwrite=True)


try:
    with siril.image_lock():
        data = siril.get_image().data
        header = siril.get_image_fits_header()
except NoImageError:
    siril.error_messagebox("Please load an image before running this script.")
    exit(1)

_, IMAGE_FILE = tempfile.mkstemp(suffix=".fit")
save_fits(data, IMAGE_FILE, header)

import atexit


def delete_temp_image():
    import os
    os.remove(IMAGE_FILE)
    siril.reset_progress()


#atexit.register(delete_temp_image)


def get_runtime_version():
    import platform
    if platform.system() == "Linux":
        if platform.machine() == "x86_64":
            return "duosplit-x86_64-unknown-linux-gnu"
        elif platform.machine() == "aarch64":
            return "duosplit-aarch64-unknown-linux-gnu"
    elif platform.system() == "Windows" and '64' in platform.machine():
        return "duosplit-x86_64-unknown-linux-gnu.exe"
    siril.error_messagebox(
        "This script only provides a runtime for Linux x86_64, Linux ARM64, and Windows x86_64 systems. "
        "If you do not have such a system, you will need to download the runtime from "
        "'https://github.com/Seggan/duosplit', compile it, and place the compiled executable in "
        f"'{siril.get_siril_userdatadir()}' as 'duosplit'."
    )
    exit(1)


import requests


def get_latest_release():
    api_url = "https://api.github.com/repos/Seggan/duosplit/releases/latest"
    response = requests.get(api_url)
    response.raise_for_status()
    release_info = response.json()
    return release_info


def get_runtime_asset(error):
    release_info = get_latest_release()
    assets = release_info.get("assets", [])
    runtime_version = get_runtime_version()
    for asset in assets:
        if asset["name"] == runtime_version:
            return asset
    else:
        if error:
            siril.error_messagebox(
                "Could not find a suitable duosplit runtime for your system in the latest release."
            )
            exit(1)
        return None


from pathlib import Path

RUNTIME_PATH = Path(siril.get_siril_userdatadir()) / "duosplit"
if not RUNTIME_PATH.exists():
    siril.log("Duosplit runtime not found. Downloading...")
    try:
        asset = get_runtime_asset(error=True)
    except requests.exceptions.ConnectionError:
        siril.error_messagebox("Could not download runtime. Check your internet connection")
        exit(1)
    download_url = asset["browser_download_url"]
    download_with_progress(
        siril,
        download_url,
        str(RUNTIME_PATH)
    )

from hashlib import sha256

try:
    asset = get_runtime_asset(error=False)
except requests.exceptions.ConnectionError:
    siril.log("Could not check for updates", color=LogColor.SALMON)
    asset = None
if asset:
    current_hash = "sha256:" + sha256(RUNTIME_PATH.read_bytes()).hexdigest()
    expected_hash = asset["digest"]
    if current_hash != expected_hash:
        siril.log("A new version of duosplit is available. Downloading update...")
        download_url = asset["browser_download_url"]
        download_with_progress(
            siril,
            download_url,
            str(RUNTIME_PATH)
        )
    else:
        siril.log("Duosplit runtime is up to date.")

import stat

RUNTIME_PATH.chmod(RUNTIME_PATH.stat().st_mode | stat.S_IEXEC)

import json

CAMERAS_FILE = Path(siril.get_siril_configdir()) / "duosplit_cameras.json"
if not CAMERAS_FILE.exists():
    with open(CAMERAS_FILE, "w") as f:
        json.dump({}, f)

with open(CAMERAS_FILE, "r") as f:
    cameras = json.load(f)

from dataclasses import dataclass


@dataclass
class Parameters:
    qe_r_ha: float
    qe_r_oiii: float
    qe_g_ha: float
    qe_g_oiii: float
    qe_b_ha: float
    qe_b_oiii: float
    population: int
    generations: int
    elitism: int
    initial_std_dev: float
    decay_rate: float


def stdout_capture(stream, gens):
    for line in iter(stream.readline, ''):
        line = line.strip()
        siril.log(line, color=LogColor.BLUE)
        if line.startswith("Generation "):
            split = line.split(' ')
            gen = int(split[1][:-1])
            siril.update_progress("Running genetic algorithm...", (gen + 1) / gens)
        else:
            siril.update_progress("Processing...", -1)
    stream.close()


def stderr_capture(stream, buffer):
    for line in iter(stream.readline, ''):
        buffer.write(line)
        siril.log(line.strip(), color=LogColor.RED)
    stream.close()


def run_duosplit(parameters: Parameters):
    import subprocess
    import os

    env = os.environ.copy()
    if "WAYLAND_DISPLAY" in env:
        # WGPU is wonky with Wayland + Vulkan, so force it to use OpenGL instead
        env["WGPU_BACKEND"] = "gl"

    args = [
        str(RUNTIME_PATH),
        "--qrh", str(parameters.qe_r_ha),
        "--qro", str(parameters.qe_r_oiii),
        "--qgh", str(parameters.qe_g_ha),
        "--qgo", str(parameters.qe_g_oiii),
        "--qbh", str(parameters.qe_b_ha),
        "--qbo", str(parameters.qe_b_oiii),
        "--population-size", str(parameters.population),
        "--generations", str(parameters.generations),
        "--elitism", str(parameters.elitism),
        "--initial-std", str(parameters.initial_std_dev),
        "--decay-rate", str(parameters.decay_rate),
        "--output", siril.get_siril_wd(),
        str(IMAGE_FILE)
    ]

    siril.log("Running " + " ".join(args), color=LogColor.BLUE)

    proc = subprocess.Popen(
        args=args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True
    )

    import threading
    from io import StringIO
    stderr_buffer = StringIO()

    stdout_thread = threading.Thread(target=stdout_capture, args=(proc.stdout, parameters.generations))
    stderr_thread = threading.Thread(target=stderr_capture, args=(proc.stderr, stderr_buffer))

    stdout_thread.start()
    stderr_thread.start()

    proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    if proc.returncode != 0:
        if proc.returncode == 1:
            # User error
            siril.error_messagebox(stderr_buffer.getvalue())
        else:
            siril.error_messagebox(
                "Duosplit runtime crashed. See log for details and report an issue at https://github.com/Seggan/duosplit/issues")
    else:
        siril.info_messagebox("Duosplit completed successfully.")


ensure_installed("pyqt6")

from PyQt6.QtWidgets import QApplication, QVBoxLayout, QLineEdit, QPushButton, QFormLayout, QLabel, \
    QHBoxLayout, QWidget, QComboBox, QMessageBox
from PyQt6.QtCore import Qt, QEventLoop
from PyQt6.QtGui import QImage, QPixmap


class NewCameraDialog(QWidget):
    EXAMPLE_IMAGE = b'Qb|TeVnqM|S4BclR!}$~Rz&~+@G=6O0V*B@yaGQxp3i5$JT|qXl3dA9CxB-F#=BkTk-+`7HniJX_Fc+7ChmtuZO`n*`eXmC>+k>nRsa9=^yz-q_G_asSA28Z`}F_M_7nZ5cu$*OGv0}wt?xh%n=jmqs=qyd!~XT+i};`U{wewF`%aDdi=q;9f6VZ&flfwtf&ib>yW+LQTafMDrzNxF%qC=IC2x-~nUR#0zC6NaMp9Pz^9h+5Nn7L0CS+wLZ;vpUk(8CbJi=y1Qdao$37Hv5TjR_oWMw69k1&~$l$E|b!e&NNR`~M?nHfo2$2N=b>%dkwrAN0u%jd-Ca{8N`jm(qD)6+tdo;jHrNn7K?w!20LSXLQ!4zpCFMU1s26M8vk1*~`+Jj!`fu8nl8hEg5gz5fx{)f5FuPaMhQ<Jif-nxr?(4EC6dF3X-gqheKqd<)ZZIjo3<1d|M%3^KhE3~O|cbEGw%KfmPEWm@E^7#lDxOpg6#?;|M@6f)10LT>xj0CflDST=b7ia~FWFp+bTIFdbiBK%Pj-hkivPcYgbcKaMyy&4*va5YJ8u1qlLVp6phrp!p*{stHv!6bWYb8m*6ni6`&T|1b=j*({pNt;zRUa*kcwSPXQ)vXiVgh}tFPAl}H1GqiaQ;Qb8oDk}JdiUl`O9LKc3TU}j-WgcSN<nsq2bb@;0MK?SOS?Ci6%&4sMwT#|f25C-Jo$3SN@18x$jVDcOUaVPE!-tMb22iLx6aqF@xUOO`pp0SDFw&UMsLYNS~_1%R>mw_<HVguhi9YTiKu@n9&>0?^)9~@0d4mek5Xl|BLqTunTwhVK)2pqRLg$pZ)Qy3REAyOP<RvzBBq{lk^y-8@>!K(%|+PtCR=p_uUd3(dsd8OOw;@FO`Onl1VBB5SIi}xPj|rZfsIsMpiu`+#3vTKwz$`~%(>7C5nh>S-^4l7t6~Ye*A1wIL}(T_eI+u7lAo$Ic20&u%EQ0`pOmUWJqgSdUche{32galQcg<iPn=ampq_^4988@1+9e_@`#te8acioF9U|n08Mmf_+StX8jA$E+9T?9DnPp#`^kT%$lLGlNi0s_ZICBuLK$uH7f#;`SCgxlQSNJ8~&iSL$CBr4b1`$B6TP)m)8e_KU%B4K<`emV3n8AlrI5`Rbrg1b0)CtrH)B?Px;^JMw*Qru>$00w%7VXM+A1U8UKFDSM4@PUJ_Xet?7q>-dr#Pb+z0W%3JS&<Ohpj}rK@CU!m|+Fk++0}cn0KViJnB92Qmp<pLDUII(HC4&aDD8bIR3qp9$v~w>u$uwxnZ``=WRp}+$7Y*eeK`=(PhbY)V&UmVv2UGU}s)Nyd&ZD**@QO@6ENB02x{HP}8ho+m{sSM%;GdZ=bzX?BZw>os*}~Ql&15cL!dcDc<$mBBI&g#0iUZmHs>8$Q93ui1;jugODkI1%>q;5yEu<`1!$r?RDFjVvvOz=?8~@ItuPNcX?jz_)|s~h-J+rXc7O=s0DdX#l)q9uTrG%l4k3CZzpkZgMSiDS>jaDtE%8-eBwxyv>7YvU)Mi@s5i(NKR-5U(&~)DM|8bNIrwFVn_MR;-mFML(6ASHT1nsJsyJQIg=&!|6G`9JpshPe8U)t@GeM#7Vc3+N-9YRRv|=AicsOQWrf(3fK9cllju9PhbPn3Axnt*2el(9z)PL<R>XKB<A!a->TryI=7o~1{E(!6wvxJ-H7-pZCL4xD61_5H7@=d}`!cD?W!cD=*NRGPquTrDem`f-p*hTm1!j0vzmH1Y!eZxA87nO)ur+Df-W6)k)S~hH^oF5?UjNh+W=x=C}{Jn3gS5*UdS^4VVX{J2w={N(B!QzZkWZB@A36{Wyt5{IjR<Juf8@gBK_qV~nxqaZEDZ%|xli?VGc7T~7=Q$4eU0KDb6Q~njro36h>~)Vd#nyawN=TBf+4eGqRIRoSgm#3Rgm#3RgqoO}gqwt$OyX~E7A;o>yCpY0ugcLF=6x~-A)xyHPdBFoz&&{-@=vW5=Yh2?l&}TrJjQN;&fDD~{g&MU^?)WpnT&lc2r|~x4P~-shilC#9!})?P>yE_HwiZhHwiT{HwiYL{6Q-g?Z`~tH-P*%l-3lcSD^-8?a|aQ3KPk(vFlHLXaV!*ZY;E=F<)5TK1pS+76uI!^jCociw&c`e_&wr6F!j4dj-Ra7I7T~ch)>HZzr9?&`9^>dUf>io6$lYbQNq3LG$-`Drq$%sfclkFro|^jTZGNf<;Ds0y7p4ey}s0Is5azoR)&w;g>Tsy%6XoEgL||s}x3*bR4DRNB}UBiC!HZ2G(Hux}SphG8lFXZY!npgm#3Re!GO5gqwt$gj8ER5|(5z^c7@=b=}{q_2dB^u>q2(^<e5El*e)Q57HoBF}R({)CrM1N%%WyNW3CweVQ<-2^+H#s`)0SBo-XJU`l+`SN}NAp-?AKCr}LT^>xI%fG<*cSjBi(%C!?981p@wesYr}Tt{ZV2mEZA^1LiN2?oi|cA9x9Ef%|;MdLyB1X~=Wf`Oi(+Rf(C-`?%CsXcm>LIm((2A~rb^lw0rs??5xS#S#MEtm}T_elyxYg}p}A-}nLtBKy6fg=$hLzzwYs}}c}5~08RXv@#N056@rn+_e_oFv)Jf$)4cnnb~?ZJ={^ezx?4=eqBbZTjrqv-o!b+(n<!&)6`_nn=(j04Go<P$T~(bimoFw6iTI<l!g(KvV+60DRbp=4((r;rg?e?~u6^uW#bn!wzf{0|bds$a{q3>dBK2Wc5tp9iOh$BOk1=WK^A#xSayfsdp0D#0>&<0(AiQ+}^Ie=0vFV<`X5xB0)D0o1+-+V4lA$lAaciPtpu4>7cc8n{UmdqR@=R;L9MTJ4_-@vaHj0@PR*hRi2lpig(F2{dW4c6O;veESh}g2!4-qS!W*P>oAM5boeICc)Tn@z$oORH`f^_VQlO295(F_z5dqJy;A`dhn39%oP0%-uHkJG;CLsqZ$+x{vOQ!{Sw@!azpet-{(cjqQBvoj5Rz;3o6UDyS;D%dN#7edM0MO9wsMMsWF7ezaM?<Gke6_E*p!{}O{yFq++?v-1K6*Ib>GbP4xxrcPi?*bvOzfsMD^I)O0nQvLgKaiiz{XRIX#aFwTrv{b$pVnkWNQZIr(Tgs=*91=1|IMAXLr6bEz=5jMU+nV4@G`Ehkf_Lm`>*L0)&Xd&~{T6Bg|K@)W7;DeX!dBjPwNOM=Gl*K9|l0?Ng_jbF2bZ;Ezp%U$vNa0ON4N%f*kg!-HxRH-}Ub(oUJ{3hYGaaj^HPejF#P<;lD0bcxu9btt{nnDIcPm?E^6%0M;XOhLu90|_J)CtrH$>gz#ko8%F9+k{N6&Vq6dqf1u7YiUrQkh6%;Wp(+b)O-{Z&LTrVC2#|byuYdIGun1N66FU*Qq{w@}Ch+1zgep?c?F<8IHf^Z$h$1ux4F+4U<yV<vnmv&qV={OP*N`a$f5aNZWePcB$`U`ARP3_FF>+w+j%87gEPqM%YXOFw&~NY5vqzwLnYlhZOxEFAjAAKlqulrwDF~7FM1(RxPS#(@Gy{G7^yrj!i3wcod7|esvf^9-KzE`|+Y~Z5eFCQTZLvROCw{MyU*rR~8fPz5rhh>+3H`N3S?A-~U^0gHtFgI%ys+5$=SrX_=tGo`9_2g(k=t>mqrP3hLnau{*~Y&$P-%s~5+zN}WHCq|Bt14GwHb$E$-e*v#@ApbZZ-4inMV+{+=J@2uHIm@u@HzOBbv6!-a+8=+d}NB-@nm9^dKeb6{$xM9%o&fXMA&)C6Diehm+yh|4|p?u>v6Q~oXi9o3XY~uTu8`#@u8<c8g?Iqrk(rm?8{s8>3m%lx@8K27{@cr%ydm&-Dn#haDeqU;QIlL@Dtuqs4INjkzr(%5qk-A<nnbBxXVFb#Gn7+9q!9ewf*)O*}$>R^OW%Goa=NCQF{0o|m)}H7}4NV%UP8P-e@=d}gk}24*wvS##veB|;vVH%!waDjxtuO9eGwf;YH%w!;mB6SW3pcfTn>}q1jqv{}$}uaBh~La)R2CGp0(wdiuFqvphul(}Z0_=bgH<yN%jj(^JaZ*p8?kgq&rurKy%903&O3aRH8Xd$tC^$1EJBir%@pxcjBXkBi-Cim*no3f{?T^@2yMLu)QS><n9<ZaNbN2ntJk8G<=B$<4|!Ph9jXl<TLi~3L!X^4OGzy!SORqdWVJL5_Mf=uNw@2%hgK7x<1%3EM-x&+EN2m(pPWS82|}o8e;C6&XM6NUa$Q-5(Fnlpe{RjEy5#p%gi_%@M^Gl^%|g4vQHS)sSV8YX7=t@0E`u&osoF>n^-TH*afV#ch<`q4N{F%)@^u10a{!Vs%Y|L*-w)!0M+q$!DR>K22o>MvDJl6?iyiUrEgIWg&oW+L++ZlMi;krLFKf|i^9Lb6?6dSa-=UtQw#|RW>TXOB*H&jduQDh)8FrRf5W_>@z8J@1_5M4jR=THb#|And-dUH*yA+6;i5ek`ojCX?M3w65soF@;5$33|;kY2emjCmhH2&7KT?qLfJu_^=S;d0S?E__52{%qebu3LEq^qaEA=cx%#L+M;ryUfnv?nnfpWwD-lzTr`?7I@6yvy||qzjXi@Vo&U>MYwW`+68Csr+Cw;T!Liotz}uww~$qacaM^0H^!}nex_P%n#<>r0$N&#pl<-U2*02a!qoT-w4(kl2k1bh883zzn5^XsPUE&RYKB6fjWU7{R({wfi%?9JMu+478$y}37t=pQp-9^;+_vrGpz}zm<8qCm{fuuc<K~3I`__8tkZYT<K}Shy`MlLYz2wY9lp8_3#$CibB~Lne*j@b8%H<E88RRGH`M>(C<^nI+e>o=5nr)QIF!1E<A;FBIaXacm~PSVc40|G=82%PGwcybXm{jld{7psH4${Us~NjJR4MblWVV0jddT{xdZ&8G(dq%iUk#F#uV4&V#e56xnM;<Y8|GMkuqfthFMV0#fWxxbdws!-l6rgF`7Z9PF1X?#AO4rHGMl8Q(ggKdh_R)UB!8BWVN`mCP^`bD5*)pf&6-^}!)AdHicK7`_6fjXnDq~CLK3#Ij5EF@A^0N}a7>?+JjJ;|{DACxf)EELtTlq+up+vmxl#l$xsJ-FtPnpB9~&GyQdAs)JH1{np?;itKdh7=f|4OIK^G#T-4@4f13a(lJq}ZiPgk5`v*nDR3FP;%0BGXYeBAk0Hj<ygIGTqAPa_k+0+pS7<O2}Qwqban3kZ7j)<X>?t|MTkX!Yv99q4^($=@f%8|ud!hJhW58-^5j+}P7}FnhzBSP*DCEflK51}5*T#0nexqI^aYforT7nbWYnuWL<$H?WF!!UGC9D>e}GQH|DspyBJ?w*HJk!b`%%$-z6fS+k)@qZhNNCK2S2!acK-`(q}}9a&iqSvcoLky8l+oaqPThhhKbeY(AlKJI*+1jQPC4dRemz~j)9lDgB#A22}qZ+t9%D>8ak|KmO=ql!Uqk2W*^6k@@vvWgYm@LN28r_G~=+fN{EwDJbqX?S9-UG?PfI|-5_^hI>-lr`}ymW8k0=F(7l@o0oZW*ATS)XFv;U(3D8l8&T*qH`R~jng?^$pwRTuZ39dzlDLbm$*T2{B!*gIu&~ImDOZR7_Wex>jx!ur;s+;I#LK>Tl09N4od4!AZ@hr2HQ^{ZL;!HH~qyy0nDH+_5nPDVtwhHq}Iz5={vINdCe2oBeR>@duz9fL2BxVJLb{`+fN{EwDJeUL4CV~qH?tEwDJbqPatiy@(0WjlmZLw%V+=bNG)Aa2YlK<+iBztww^$M0RG*{0000000000000000010!Z`rX>1g1V?&>B`gXSW?!4U7~;SZ`y}w(HSa(ov&40^zJTrb7Kl44u3cj{47r3+g4@<)D8Umt>1HodlBQt-dR=oa4GnD(ln1yym+ItuP2IKvK1)ghP*J)hLLa(1Xy4o5ocb{oJmz#9Y^T%hmoE!I2|zqo?~4qE<8Pk900TSULbT>B+N*&%^MqWwko`(QjkE_dS_KV`tu>BJ$xQOQ)T8xk)jOnMj~*2zg7e%)*X~L-RJ)oxQJly5&H%bbRYqxEc`c2>`42YKH2?GPcC~GZ_JlgJ{At4VIKj6X6C+6VX=Sj>pe_#%8DvNq3$Eqbi$;y#DH35DlyrT78q5(M|6n15M9}FWY<c2>z!DQ_lv}tb>Rg@6+~172ofZJ75&&>rz2nLu&naW1^hu@g>5zks8jiIc$!kXNm97SX|=0LdEKTt#;e3Iv|XcC6$P7M;^_l-Q9}z-x^SMuRr6g{y(wn6uVSc&UI63oP=ht+nl;lM9aGY*G})-S3GqasnV&4Nji550C<10EztoG%Yc?~92rF+;57=KK+6Uo=n(^14l<ZO4YARJH;Fr*u1w>j+Yp~Z<aXPm<I#N}si)__PJyseW&12R5)_W*9eN4SPs*t*^4eQZ^IJ4XasP8pEq~<1?Qv;G?FN+{^m?-|w+7lW^A4wYe~uDaW6>+77ZIJ@mr!v_aHb<S9PpEpkwsvvLsJ#vLsMB4yL?f1)54p)LW~lLB$}*mkPwfI%)#c;VvjGQKN_2AM21MDNk1{Zsqi7gd`9tY7&k57&6+r={%@P&3iNY&GJGiA=V6)XB4SWT?gruwSmML?eXqxZB*`ul=APLZI*w-*SLi2k5uH&%z~ImM?f2<I%N}Nh0)D}4wiSt^oKgTw+e$fAZGvgs8SZ@~$i@cwCd*oDMq9~O9L;)n#P+b+6KF~@*#a)KJ8OWl!>nA(1HvNgdbV}E*#o{`WMTGkd2Joh#dnO=zhMJe^Wy~P7OHt#_;p~;F7^Gk4HN1z9MO{oWY#40*1r4^v}4uR(4O>L-R%krt<B!BvNL}3G*11`2bXD3{`2bqKy%1gru%RebR$2C3j&n$a;zYj^*z@wnw+L?#3zKl>DA?z^b<=5&$|JAb}ym6^H}2BMmWXxS59G*=(A}*q!AdKc#4(PZ5R`}jt%z*=mJ4!=`ATPiSUv?G%dM#h!9CY&}qp}!$QfzRvkq+{jb|<pFp9YHP99p@q;0hNwX0FCLw>C-gfkd&SQqa>^Lb($CU_^Wgn(jDNMC;jfKX4H{<HXNQ9rfj1`?0Vk(0|MJx?ELi$GH=AnarYDE<zi3P2#pt3~ULzn@Q*hB_hAlLt*OcD0Jj%G|e1T-Kl!BTrXan^w;w)HLWK0#PhqPm&LSacl8bGJ)CeA{X}v;=i%J+!dqNgt}c&~bZhYP*+2uvI}t2><2q>9VW2o?vlgF!pUhQkjI$aZtX)`&{4gIxb6CbjQbD|HsLwxDd$P|IASj7SP&h%=N*>jZXsRrm5FE@6?w3zz^;LE~~7Pxb{=~n>rFVj5LwQD^-6(y88tf;DFRiq`0)6f!1L3qZ)Ejw+S?uX5c7Kv_zhgx$$2)!|G*htW<ap-QqUm+Y?)gkX=MTSF{c(C*vvdKyR&J?Svz*_!%tCe~<tG000000000003F!^6pZ%J*-ZD&qNs+|?&N|pFXZFuD$$ZNR~+ayzB-<>mx=`V=4-2**%r4jrq3e{o02OaEOjI9-evR2)RsFZM`2%JJyOA7|B_TiH9Up9*baHzg;L_JC<&KPR3vzrOg1qNO|i2xJoW+yUEq<k;dxXQi;a+T*u&o(5?kcYeH+5NKHv3~E$s8aRg~|+<Ghf}QH5lLAHKst0e>OZU=jRUjbq)vt|}e|2KjwZfTz{h;?22?Wb93EPEIqeuK}Ofn87e@*|Jb#Yl2*pI0Pr8a`;=^(b*475iaZ!!t?vmD1=utkdKj|;Xe@#=xsTzWtph`aC?v-$V2z)vc1<CEg}V=(l8^6qvJF`iaj?2e(v}GdENXnr&<sQIq|~O_Z!|GRF__z_m(Rx54UY<{DtH~0?cwKIk@Z1HWP}(IrrE>jAePf`b(C81J^fZ`dSiPW%1ZLEBJi?0-KFn>!v+fEy#0_Tcy{)1vS8@isCIVs<~t*cA3i4)ij^0M+}D;L87n7N{wr&KO!k2k{)<jhEFmBNiBeQkLil4r#q+0xo<I^t2IMLTT<No5RpJ!<oIbX=Kbi}a#yb%m-B-vgRjjqp?=$TykrbDgquy^)<?@nNzqw+up571Mc9?0b&iB6`=zI&dXT?y1DmsrwM&-GT+MYUP;NX4Y7*;(fpd!~t)y6JX&@W<bX9sFP|S!VByqHs=;NjTl>=W+3^<3a>PHqMDdty%l;`Ld@(!<@p?wT6?ir+Am#{}71gIq_C)V8w+~?;MZ^^8}Wpdp{g|zA;Rd#f)*)YYgl=@OOS|?>lEhqIgZQ%Yfh$v}k4+z~?xMhAv3zg1e@ib28u4ZUrxf$smK5c)RTt7$$0Jn@6dC32FblxrP8X*%D-z9j&i$>@@JhM7=!jP!18=0)UTL%+$kS4j;g?wD2M?gCkR3P5HzXs%e=>TA}YG->{UW{N!bILJGH<t?g-gc2jA}~tI_fG9d3^{>+>~OkXzD;+Am${znBY`!EH6IGOcTg}m!el^UAWZ#{xSdIjr~1z-a{HqHkeraM3%Hx&MK=fOmB7B~JsoN|<af2+Epb9XD%G)p(}tFgBAjX2sE~X<4&*2URi9OxGOsOZF569oMplX*{a{2z@NgTZTBmirmGk0N%rewyoz2UN>ya`3P5l3`h%0x}FRWJh0xyWqs!@Zm4Fnv(6dNJ%_<M5UrRPhw(+Ug|qy>-lV20;M{t&{oj?RxLxZ%l7E#F{B)<z<mN5x3-5uqPT^crS{i$}_Lo=O)zxon`j);*PGu>nizF9re$c8hsk-QGJvOF^(#nfvUGAkk9K@mWTV=zIpsHFgDvwOx=ccb)FFREkKsrHIHq<k?tWmqa$Heb%5cx&Y=zYnI#PqE73#o3UsrT8H$sQ-V0i#|~^vy=bnVc`Vh6uV~LVP4&jsI-ysJ01_q;$3A`prut_Xyq{XUZnl_Q!WuZDYOrlfl>-m&m!HmvA*ddBD?3u9Y5UAMPWcY%5ySUMfMu6PDSbBT%S-6C5dhkog07aKHQJ_oxIVQhwi)Kmb$_K$5NgT6W-gWWQ*^_6g^q}{$~jjnLnE;v1mHV*-r$jHM?dDF{2ZH5gHXjKG!Y6#3Z~3ovl<{H<-_#lQ`%t{U}|S(%3-QZfD8~zE6h@VJUtBvc4&Jwvo@!#Ei+;2;-j5`Dg&X=K{)2q;}Jv&9Ow2S7P`*4*RB%Sbvu$_m`l>4YU!qn?s!Ez=qFE{O>s|;bsPms3Kty5Vc^5h0c&W5&Y{P|<b`;SXgK4i$p86>4PJ}8mSLbsZT7Qro%qGHLJ9Y?m@S)+5KIJ^7fdZj%n>CuT0h(uOg^(zm@zda2z!|mZsNp|K#M$*Sd<+b6}%6R5KV}QU@5CPeJ#}Fe}l&+ZU4jPhkA_vDOr0yg^qbv2UpKYN-++RVw0-^MLxLZid@nKE?ah#%a$X%f$nSEnLB(x|4yh@WLqTIU)OsQTPcQ)8g=EuVk`x;{>Z!Up|GWsuzrlNz2jq?rhw&(7oByz88a|iGL-&!15dQVv<7t&o1Rm!Xm^zOjFYTiqf$tfWA=U8W0yG0u&Y0)q$R*(7#h(Nch~~pY3#9t=k=GBq0NAv%B6keaRK<}6@v3<>ot?8f8()IZ#g?DiZS5<_C%1bHqc<&oo1E7DLZyj4PH&NQW6@<-JLB2JR78>>^Y{HzI78Ci{HeeF~7%Xpac;FEcr+_R_iLYeU(PqbMNQ&+i;JJYc9H~?8L>qtroPUg!<aP&MWvRf(igrpMis=9&}rWNP+0F3Qd+yh`5{x#kpZyUZ7xaFVGtZfYD-}L@K-q&H(&<{GAq5UQSd9f0IBZUuoFei!fUYyYaKBz<{ctu)Ka+|A$z>D90l?lLSvT7Yp`mPGckPymX<T)EuQDgRdRt8I!G=3>Zv*+#qAD78&C(B(@y5Z|Bw8*gY=cRUtZ^M(=sZ8Tw^E6TibBHals*y>zu?Z=ll7c{yck?%?WX@$3BUW96ex*8?NaO0pqva9y0bF*n9tB>>cKl!--f$5U)GeGGK5mN6QnIA>~k#9hWofVaSq@nAqLeBdcxFprepEJ@VaUnaO*x@3$L7KJ9;#u#&!RB4`{aqv&CjljTC<}{r1zT<V+ET3zYhUMICFBei@3J{<H*{Gfv`69&FmXthT%Z7V^1zk2J8Von!(=^L1@FO75`CaskcW4LX<KB-@gpe3Y4#*St;L)zlgv%CAOepl1bYegR6BfXJG1CNUg_W5id*~*Ak0p+BtTw*i|6*5tZU0bIzvieNY+$hSV1I7r=!DUFe4Xe(b#5j#=wbMJ1pA_D`XZZcYVric!y(^}_|^FU$u*#j9_Gp8yTk(*(aR2LI#-mm`JJ*}^rdJS7#LAP2zB)*v_}9CW^DXI@FRA=t=LA{yd^(<!&B+q082#zF#xo?TlkdCCi^^^QcacC-x-k-_!Ss#@kKMM6Vftk%uzEQ9RgYca6X#Qq2#-Kc5KPA)i)t$!7rt6)wYz=M38>G<BF;I6w~$9soNWhRBLT7LH<7^5xKX7B&^C3fjux&xIysdB!+D~Af)=G_?Z3wsQYW(JW<m!<eXP`3L-RHHAWw2J(CJ7@Lg`2%ej<-3}Jy3w?>trPrBVwmj6l9T@%dEwSbzgSQ8NQ8M{=>RA+?;XldzdOfhcJGLv3~Qvs!9zDVHLWkai?u_Fwmm2XS3lp;TyU$Idvd<N>N1i5PcW`UVUN7a-m;WI{TqTBvYH5{CcKCDk`nF0YlK4=RiK3+($IbDGp3gSs00uKfZKJ|}s7uq24xZG>M_{iD$4POS_j9jyM<i9wz7R@rFl)+1mdMszmPmG*Pc!o@%Br0!`Km75n>>t^@okLxOu1^V4lvp3IDiKZ72&ghg<dSq&5V$G0K<vxz^>nT-_&{mqJ9$R;i`AXLvjCASuEW-gv~m~$yF*XqqUK8XA2-aR(_w@eCaOl_2`E3RztYP?H{>Ctmr|CcHkm-kx`!LVyw_}P@HQ=bFp6l;XOJPT#&zkv;T;%-yDJ<!h6;<hOqZeq)?*^pb}R)%vl@JXIqn@vOq9#F#0U7@*kXjDYkw)psymJ2$y9%pTn}*4Cu6_LP|YzO|M31ZnwHndg4!!aZMe#qqWh|#u@C3vyf3yn^P9cp2S&40ny^^o)hLoKBYD6?EA2)my(^-xx6yao+RM{t2nn>*tgRe<X2~})fo0dUoKK5vzU6Q8c2gP2=Tr{qJhjxP&TDY052Vx6YtU?hOhv;$AshSFx9(V`lw}G!A7O5YYF!1k?xmm=8L6}`1UNFkadB7|2P)C}beb*}m=K8Gw+GrRWlm@e@V8y-4?;J%-gR=Tqd@O6=ZcK(XC8CkO3sbee;5RTt_d!?`%R8$Y?Tbi*!Xp7b$XVWa&-VL?Oh`)uJbbhXx9HmJy?+S$folQ$C_5JR2~v3G*<^*f<}<y*qmSm4N7A5fC~~o=}OI{_RilO1FbsMVb`)Jj*&4%_pO#Veo0s+G7346w!k@&ht%GDb)#T(4YB`UpE&E%pmJ~XW6z-ykLfvgGeiO9El31svgD0qnX&QBozLws@u%DBI9L;tB=zLy6VB#PWtGVi#c8MqE~^$WR#<aSpR#5ju|kqPg^saO%lKXih+euoGco|-Ot^vuC0RF-jR2SCBC6%P?{sWk$6OX>v{Ybvl+24TV)X{9l6A7U6*g3zKPadzJ{QNylfEozHfnNGj3KMlaL1FO03+UIy7+N*G+7qahuYAW@uMOVjFF!l9A|)34Mwmf)WPW(XOET4d2Ami^29QIcWktRz54LqkI=h=mC<@oh8lB(_n9Mfgh92ns4Q9I;D8CRPw<&DUvS&jRF1IvI?v=kYV=sWRc%R42tiE5CQ)o2vra7<pNR*x;OtC#1UUdex(DxfxU0o+vma}B^lRC^U-=w8l~-JlC}}uvV678|X%{lw<jDUqKxcct^0YF)#|~<SF`IS*8lBk5!keyP=3nyx60|T)lr6{dv+G{-=a(pEE1zfyb#fYm@w_wA-XUek57#oYTEWz$9v%n?kkD~u=`REo*4xMXbgF>KFB^~^hC;zo`eL3<o=WpCUP#2_;hWOP;tZa5W!_C>C}Vt#6Gl%cf*=u|A`!C~W6`<!wF*bxk3x3ulVspNJ4>#gb1u5RxZc8sZ@Qm1dQ_>-|JomME2%peL}ax)ALt~wZ;H2Kb<=aByG}=ts=DC1sz2YlCJ}7;PCDs!{pQDGdd)<6ws(yw6PNLVCd$H~lBDdOc=Q9%(_m@Au2}p${|K#|4xJ|#Vl=R%_L~G7VQ}hy#w(Zl=pQRdA8t*yd;dT{v1ytw9}CJ6e!D-$Q%-qqO7-vgG}h)1JW(DSc&n1VE`_oO_-cI|YJc1L56M)nBaLx}NIj7)-y~w!--4P|ko+4b#LV*sDnP+H(Xj@Nz0Rn;{lM*V{Dml^3rT>3aA=Ridw%T(ZkUQx);#4Bg%*SE4y5<84^YYF28r-}pvRiDq9Z=NnW%7nlUb1BCK$bK12Qi;p*Tj%hd2$kTi<4Wi=3ED_glzyWk$r5!zseCH}F!02h(22bBc!;k(brRR4#u3DS-JiqDxY9Xk#Wmq^%~wrwGQ@)x!;?UKHW3_sgn;R9uo?i>+R>ducrkWH1Tk6VG96?g~(67iH8|?`OdtxY8J|K?OI0pg17HuSgZwB;hXd5g>ThIf9O4ETAo`koS@kVs?pQ71}!b{*pJ4z-8ai1`8bPxXHuts6KGKcu_uH<M<gIo2vu4uM4NB5Q$|2bfietcA4lClu$FFg=pRqGFH?$B%!y+DaoornzqftcN$dt0Yoe^pPLt!-R(;!!kTn7^=%Cgp|1u9Qo>ec;PKx>%#em4j?iouyOs-DzT~n4=Q%@+yd`XOfo^UqS<1hKD{B#p?JKR`tWZ9)2WqXy0fcos&vAHD^5oe^Y&DC$emm}Dl1SVBsMbgUUD~W4iOFVI$cUxre>T3v|A{yda*d4<m$UlN71r6?=C>86f)79<F&XtsYNrod`JVJrE=WI5|G%cg(v%GYYFXUul=WpZ(6{TWossOrHR4jl4t6maq#O?%1bT|KXvi6{E@f%`$0rpPynu2q>ar+Gwc{0@nm+!(q{22vZ3)`!no>sKP6Mr(u7O})^}`C;uq2{-D*bVSAi0vIZP;jTEfbbk3dpb<5xZa-nm<IP5x#TulP=tSHUYu6i%WV9CpBvWF|x_gJLislJ%ut>JzGW$`{5ti6(gF-pTr9p+}A!kS>wHR?otP6CCWr<ch4qLv{A;E)3us42X+iFJTzr24<<ac?&|&5&%IuZY~eNCl9Iq>Md^&>i!8~>LjAjNX)u7ksO}8o)tyleO4GcP@dx+@;SA)4nk@X)NX{VaV=O#e!&BZH&vz67qpU8B;;Z$s0b|g2Qgkg`Hv1~t*ZC3m(eIkGC3JSzg@?qR;TIi@HG2fjgy+Wq2qEtKcsBc`^KXV6HAmrP+x+v}LMgRSamRaJ<(OdPngu-gYV9IlJ}G#}>|SIo0PPptkGKnoSNJNr1!aRKicm3FdrLrkn}%a@%1<}T6NE7Nl*oE!U5Tn0c@;{5EQD5FQQict{Xn3#rMo+wb$S%J;M1725D%Z&Rr;W-q|#%Z`qj@;_!Ll@C(42#6F7aM1Dz3A?d>t&qYrV*eCE#KE~US2Z$)kLb4xu;w1|h>+`5__qTUEPVIK1o4q_@~IEy9>nUAO5jJ#98v&|D2#!6mhZ;@7Y<{X-{LStCghn_)3+Gb#$ARZLk3vQTo9Fhd5Tbg}Md@dd(T+m8D&Wz<<+AUvWxwwiSw#-2k*=Crxlq_)E3^Q#&->d2Ml-^pZ={wFVba3eHZac<-L867huU>L!cvlV7Jbl2AVL1#a2(fAJhUl7>ToLozY&a9xXt_gu6U&4R?MH{Pk5hsage0<lEhjcQDa2uZbYj+&udMCpzdm<<`3r1mZLABG=EQp_QLPju<&eA6{HelrUU2}Kb8d<NvO`;c?IRnWfB0)&J;X0N5FU>PmQdwbA4PN(SF&m)?hM{$+kg<y3;)9H+fynQS47gMEP}YY99;4ry2j}0tWI&8#}m(-XRFmc(IBsXI>@9|M<%LL&oZW0?EK$dmX{_RNuT0|u5+|VEzquZ7knFvHYYY^c_Z;C-b^|Nd!daG;iO63fv02RyepsgEwf2k%gLV^Nq~Od>oqJZMdO92=H8oq5#&qul2U8SLc-xLscBm}e%nmX9TF-C5DCg~19WJEtPeg?4bw`1Qmja!9WsC@yvC~d5Tp@@!DBve?fjG@UdVD%>g_@+iiA5WvQl4NUV*4>og1`0A#)+<W5=RGyfX)oU&l-}-g))9>ZmH0nYFq;(NN#(>BD*Xu*kW^a*Wq)^jMTLVd{F;C@{HHpbp0~y_0>!YDTs0vTNiT@HoO9RsbM{4=0Cew#y@}d%Dz(n{iqbsm#;+ZXCF<`X=~cn4&Y(le6>CwQ`)%FO>c}L}nG`uDV#vRb<31eIFxV=s|g@r0HqG%}to?yCSh^6-W%s^d3CDTXwn_rMtckx4zOVK%I3Tqax0t5F=T~)9zv%>PvY)+1frd^3}7>z-{mOdzS5BfwpY4Z}Lsp%cNyQIKA*C$PuT}ShOG4)q&MO>5}F1Os&<<4q~O!wp{Z+PIGnD4Eh7c2p3^HpeH|o*j2pNR<5G@Z&mwt8AtK|-#eQAl-Qo)s;79!U*#gZP9&uvhDQI@k26H?K)j2qs3NIe<GMP%Z$ckCDa$5YO}sUrj$_1eESse}4(RtnN~|*J*<*^?YC-dQ5v4iv%Z5aYmw|6{9Tduc<<ADSutRP#6%?gzC!scVYyMRs(DbEX2@ZG-Oh4=Df`k~iMT;#gUQMWeFYEk^_DJTqq(>c6k#Me#K}YF`6)8HISQGCb&5cEP<3C?TL-!HL!&+laIRDuGK*R+NZt7pb7Zr<Fk6*KLzNUw+@%_=q-=zxv2u`$e<nt#(qAhq>CN@V_uL=)2TxD^vU(^TP-}=@vB^vbl=9wFZA7zM>tMTpp_bQL~eYwP`7sLFlgoPlY<JR#qu8AHGW~$2BnxBOkh@msOiUJ8z8o5uko2s<tfnX|cUQZ#kb$&t-`x+dc{5)gN8Gu<w$jb%j@JWK=R`Q~WN|dG-AR|185C|BO5<jnQs^y~H?nMQ6G<(*i;xkig$RQZc3%*dHW)xe(yh(jIuwi6#_my~isr~O+W3x`Mf${4MlYTvR`P103e#IcuKLR7{1m^EQ`hMyO*(*=hsGM1k($dS6GKT4E{kZkF3cQ}Zub3R3>pN#yp0aXm73ob8lx1&5<ToOS-0K+PCWJ{?CyvBxj8)TqRO7uRtQjN+7vF?r0l2@@L~F>K&ZcU{^Sl~xHTl`-VamYG&&h<4c-Wuy>f%ttwj0CrSaYw)yZ2@Np46mHR^>4p!VE#vX1*`Ny9#o}3~GHmSt)zWaT>xEL#zW&#BR}g5h8_;E`Ie&$ySPsBAx8vYv5EJ#CwbWeZw&0%DP{2&l&q7rm>Y`^7b=Z;#`@r({^`-Q}TT^@GNU9@~b-^`7M;(Tjm+j_eGm_@+h;0Mza|WAB+i_q&N&$I5DbUr7WI7&g<rC?fwWkCIY)jO}shLQ}FJG_`1&I*IM;%e-2W>&nx&+buQxQZFrMJg|iWh-;@gNG?M+ceL;CRt~ybNCv44mXtP>jV(TP%z#UN!*{yDIHGAjxqgUm@&L<ODS;^jPSugIP0(Vp5pHErm%D2m;XkqYqm2Gl)s9uDa<FE6BMa=1Uy-I?=VmJ)apD7z~eo7Kd6jruH7uAd&`V9y{37a!mn0R1DBU1*vA+2ws?fgAI@x69rnh8TbdTz=70vIS|Frp8f{aPv6<!9go!mh^7+`x*qm1;#4kyTR=q0QKsOznVTRr7oa_lt6d3BrHfW4p+0OEG^?5${BrVpRtw=!vgv5qLUB204R+5QXq}3<3F8N;&41D(BFocA{+r<jPndHKWP7rpDK8XiTfzR8Y{+B)*XrnHl>0xYw5i$TXB{NOYdf9h~_*tV$-ALP(`ojAW-~{*n(e4;@R2$DUS~xK*azd{0yJG}vewL`r7nIRU<<LPMyb82UTFZQ`(IG7F+6<`!ziz8q{$wBdgshhiKU09#pz@rWV4HYGwoH$igoMNREr6M>q_tgVS;!ybnLz0>p9<VHf^2p}W<vQ&bbU#PDkuP?A|gNkykWxSgj2X_(oJVd)_$YnukUeKCw>Q=!zRa6=|FTj`zy0U~AXMPsp0WA4QzT73A+YuA=Y+QgCW$+Q*=xwn2Nnw2@6hi-NySh+y1J};KG(JkcST4?c&aZFxd+SJfp8L+bMg0xilq$SE*`tkiF8Q`BC%bi)9nNpElLftsk9bI&vkipfbZsZVAbgo_bqs_F)_<V1_Wyqv!kZNFNrE3uD&ldq^C#s4qqIgJeBD9auwG+OFO;4#k4RBkZa>C~IorL6(<IRbEpA^{fIMz~+p3Skh6sL|Ng!>#MhD?z4r5?idsC($PFTr+OenAL?LR?`ttGks^Kzgbx_U?6#LNRD@Wg3}jwj`?nlY#rm7Ib6S(Y~E(B^pc$b;Dmm)o`bpFEDeK?8qmSR?cmB|IdTe`Y--DRTpdsTb)ec{nG4)ks45^$(HCzjQVGbsrU>IV+rrJ1oi^mPBA#P)hLlyS6BuGv>DC)9Ghc+eIUS*vOJ)c)TteW7H&oKCHl>A&P?66~N~K73((r>N(;3Jc^GDM?8#wJ|T;d_z?-zX`)m@f(`H~2g7heTlC4`hN<(g5mgetw}cp>LE-YNI!wl6HqI^6T-$Xy^ch+5(-<DHW@wKz(fkl;O@<KG>LPXO)m}qA;UCz4eo{unQp)Y@y%gJ_-_gGNIZ*r|jvCZih!dr9MAOouUf?5&iEi=N3S(JIjiXC9xLuy8b9uA8cpl$tHeCn~C`FP;$w$00bHUSEJkWjvq8@?mKKWS4l11Hik$*|Yx;(mS34#wRbV9a=x?9N)ceQ{Zz!L3?S|^C}R7I)d6BLj<$X~ari$L_Ix#oXBxi%MEgT-CHASLWL^w4nsGhXQaYifPfQ#fckykf9w3D7vWBt<T!r~2ynrOG9t7B^6{*@`WXc9Aj`F+%J%|B$kbRMIT_J{yzP7#Ik(=w&I*+_ud|?|6wBN<`m!Yeu2RXA6EW8%=skd-v33Tl&_-LGRR7{h6mV@ivuF-%gyhQX-E@_kQC3mFYl`wP<Sof)IkCx_1U?;-u6bwpJ`4%@i0(b)qZ-AhgsP%I{#lmVd2T?5L8-U;J1u6Yld8>jh(_mBlj$;k``aJMowxdMUt2J@n22Zs9na*`==nD~O@%?aS$60J~h_uGe{)F75oEgba&gDa;Ru7GG}KsXVt0h^S~KF}G`ga}*K<&TW>xYT!VyUux+dny*@I@&y8B9y|dQAQ#}~ZqHz?N~%L`HrC^Yn7arD0r^$r=<5$O@2`Vwaq_MubTXgywBvRLub%=vMp3+ET<QH;crso+=I<a{L54ZskR5=OZ)^zzjuZTQ&aM^rD4*Tmj=De6J$GtPD0wzAABB#7<*Z2x^mgujB!Las3@rA5=lA*xq>cNs$<lZMK32GGXiLK1s<o?A1lLo(94zJS1{CDXjZ2O>f<cC~WrzmxF_2Nrg|0&V(*d*55kT9w9g6mD_kLjV`O8GMLUdX9UvO-gITYtP_j?e);ZqYalnM`ip|99-(RlXMer<fR=Gj}`(Do_Bb%a)W06E!xuT{HQ@$YKYEG+phbC1-im1dP)0C({m1JBq>p^m=JD*pi3r`*tJXa^>75^unqtZh$6A<eqg8W3G!O;k#%T1YHByt!qARmmy^@@RAYioDHtMN#DJwUyZ^9g@-6aor*CKje{mQOKSaz{f|E7A)Twssk6lH`aZ%RC?<H7CR}Jlre79cevV(TcK_n1PWfPm4MxWbjV#;3;PZ2qg4`P0hQz*Wsuy%>nPCQh-!S=s0Xo0+JZvpQQU}-HMQBTe;Fl=jGx}BE1T~O2s7ru0CEDt+F2eSgheK~6-y^Kx*#faXRBcWO7GODD<TT%W<Z_7H%r8Z1^*i8=Db}DR;|@G5Y!}lNZcOH2WikH&`I8`n&M6Suv?%vjmc}lBa8zB84v~|d$f|}D<un2W)=g0nQ)pO?a^xny*EPy*)v2QEy63uQ8Yxyw7NTZb^lIzl<^K`A^!jY80g9KM%E_Mu-pvYrFZ&~@3Q^_W?Zz~y}%?WmlQ#<250;SHlUF)gZz*wNaK>IbgnM;M6t(TS!S^>8h&b4QP13kr4RXtcon?Xr5IyL7EKyOYj1_<*4)YOLMV3VoGhg@_h8aTudqXtyaw}^)~LAevBy;sMIA(7OK)j&CIOC+KctDjRf<c!dMV?Kij6Y}SziJsmXxE)O^84bqWD^?$N6SS22A1<BdiP0#0P38P$;@CHD-hUMg%o?ciRCUym)*MI|Z%Z?<DCa71+Z6N>gPV3h+Izfl_A0?^Y<-_x}O5T|H{^jK)B2UuyHY))dGB!2A%L+F=@Iqd#p=nb&YL&j9KAjf`$Yd*$0S`aoi#Mchr!q2IDh?Z{MV7sRRse;8DXKtHv_eCSn-pV8AjVQStGE$`{kR1kB*fmT}KE<9vII^`8sctzpq*$;u46CB<%9fC2>HREQ(X~5)Gszstc6b&S1XxXK)4M*7IEM=P+b5@G~`HrHsQvIX}BxCO)3jp8TV&)v3V3yrU)xH>L64tyw&zTsy>ISkaQZHisv3$-?YwX)}?v;QEF<h}(PL^OL+rhvn{JB%Lz#^UXpH?#9=kX4kUI{*MP_t=|@OpU(uj@p6dF^A06TW{(yx@#Wu*pDZ^b_!U2=zn??Ip*$sgTT>)A7e@-luB?nR?&0ZJ$N*yO(5JS-pEtzD~}7rvi_<*0g<0ea%GgqF4~JHO4igYs}!1lDCUrOAy`M3n-Wd#i!yj`T4)+p<0`N3x<bOn2#h&sn~18AQmwXt;V4w8f@52z*yVU0f$}&+*YIy!CkKyCNl%XWk_^tj$CHcKuM|Bi_78@7K$io&I6rm8jGK)*;JzW4Hf`z!!s%8ns!h!x78$7<cdE@)QJR+!2NuY7GZTZMfWb$aFQ88$CdAlb8T(QtiFDawgRPrAdojdeX(1vA)_uJfQnC`#G(#IG;QoGd0ZdKo8w2q(HyYVCdR6~?khB5*>$M*DA>#mWU9{6D$jv8&z3NA--0o>spNjz1m|#P!sb6~)Z-!8`u6K%5lI4h^%y$sXTIadPYzV?`6<&9{#U|#9H-p-p<`*1tJrJ{RbA`bWuTDPNmkfoQ^7Z8RDREJ6H4y2_yjLT^lED2tvw2O@~kF&w$(kM!z>S7DTbd1C_lS;s~hN$K&8Ce-A$p7VBrwWS><N-U55tHCyQ@^<V8s1+h(}={<hiGU{P0o$R;-4q%K${LDfGf>l>Lp^DmmfOp4x`u#TnRz`>%;s_%mOeyy&c_E&!<!tiQ!x8_ewqe6;6Gu5MY@SL$-RK5}W;5Vg9C0Qtrc**%!=Lf|%8DPp|tswpCKo0PqTfm}H6d2Zy<E7y5O0*P~<d$yq7m(L}OA1Q3ciZ-F>Xsf26QErpMZ9)YB~DEfuJx#(Y-SSF=t`8&*`k2-kQ8Q=YLJBuN%rir*0iJ%At@BMnbC!F5>{XfNv4GD9qiNo07u_!VD5p<o2oX&h>wpgEz_e>h1a<A3qM;7_sb+=CJN>F5GFdE7RAWUslZ1~xl>Ju>9#+KubMi7m43_mQ}U?(gkJs1q&DDa*f1Wco}UBZK#}2%YsE2{h6A-bh#qW!ZSsY;eeaCB)Y6swWMcrATbKX=40lSs98?6OM8hM<)8IrLtA(4l@Nw5|7W)Xw!R^{^h_f^9S{orJ(c}T8^Y@kMO5P%aHG55#a#{j23Tgm{6BFcgjRG(Leb(*s1`ZMn*Va(WuC46G6}^;bt5)cc<ynz!Ai(FL4ZsLWNO4Z*R(mXSaKj8qgI_YwVkl=(GsGP|lVlZC+^_9cGsbZ?B~XBt>bNzmIn?s7!;2i~I=YLsWj`}~xV3EhXo|V*=I^5elRTkzIL**V4d6f)7}EzZu~sRGwJvT!Jxl^I7m#)W*)u`yG$qvyFDIp7Da7+(8I(<GWiDYw3(_+1s3i_~FFbEKn%0zv&qMD#5lu?b-QECn&q_b8PLd6$LkwMoh{C`eUoY<81T(qK5=}ukGDSL!vQg|f4cjCb>x!fjefsBttjtxKE7mOg!DhP$Z{^`6fjMGQ*4CI#jqE3CqS6$x*sot41B@fT+%jbEn4y>Ie<=RtN~+weHL=KfB1!7d@`TZO$8@36-Cj#aJI7;tDi@Fl^b?xl+R)Tfe1-aA4c1OXl-JYS!}7NoDY%sooFzqKoQm$HaXRFSDMYIAY$xsKE!D`e8E~c097NYi8(wNb#_XWWaLVhk!j9E4!C?$`)JT;}k_c1WVe~Azv@#_~y}x<ay?}Im)cXS632G-sp~iL`D)TG9ZAMDU4o0;5MpQ(Bs|M934=L3wMKzME`_qBVbG5_gI(GZS$D)hN_&pvTw!&olVkApdCKYR8GnL5r0c-3Ji5wV(RV2ye!k)IQv81J#LMNROIPkA~UG&niI$b_CCQ<+K(qF$tqjs3VP+#h{;E`ylvOb0{r;J64u6F#QjtxZiBrt%J^j*l$JDMhnj>G|Y)IOYov9Oq@GnfNY-Qj_52a^T)t3E~?UBz9uoGh>Bx6A+8T?BzWG@Mbeu4_;k{Q0r!&Xk9f5Am;_oP)?mf4DXhG5Kfv-}<C@Pur~wStE$~8f=~fIB!p@7g#atp^C|7>rA#k0^TiMy~L`5mm6}Hk&5#T<fcQ<g|<QcO6gh+LHr``mH3vK=E<*|_-Gi=Q803IOrd_lidbs`TSg-{4w>*c>8le^7WTU$iJfeU00J`_?9P`5W3Q|HXv2cTnsZJS?cD-PPFv5|_U$FNx}?4UB^d}pFQn|CH!~B!Bg*%gEn)s{NLy_MZkI9B<mr>yS2VGUzY*8xDEobsm{N#MdGcA_2O1VI4BD2s2?|6vK`OlryzHrPxv8;nTRwHtR2Og1$Tf<9cQ8*vx+TcQc!=>OIRhR+cLN;7rq^?zc~q%`{Rox2a{>4Tu=PCQ>dd?qY+&ERClu4y=*+Wxd>F30w9;Y+#)9a>&II0iZ2TJCf0wSh3}}0+G0)_E?idzs@k`3grGO|7Ezj9Q2J%vMT&|{nLBKEf2+pdxMLtiD*yC{BkI!s-=0?yumxyo&7aaBiSj>macFC~6uzCO@X6mF&LD@O;S3#K7J!|Z59P#77y2ETjc7abU&9L3yuW0%wZDI%xDvOPt3prIHiUr7M3XVgB;czKYJ4-E)%?)UT3i!0)NS_vJCIu%>V?mXU{6A&l5IOR-@c|*O$glnQQ1+)?LPv&PZ`OaX=`Xc)+w3H8Ih>HjHA8RHg*(o59icOEHp}HqU6OHB(H4x1aJ-L=vBuR2Mdtpb3jE^A`HCJnCFfW4`z|y?@6#w5*Kz+9=(D-!P5JR`ISks~m<1tO`=TyKWM$U&(exsDY0<fxDJiy-QP9F-%z*DIjM%KiB%_HjNfbhgWAxti@(_SVjg@pV_QGQnvDZ~%<^R5d1S-{ti1tbXK^z;!{_9j+UkvTW!1Hn2do63R@R{KtF2S{sQ#X8zAc7rrIweG!EO$O3A{v$MO{s&RcxLjJ0L30BG|}(;^hi|3WyPbM!}+=Ds^4(mIcvHu^!W<NZGk-1*yxQMhA{wts7ccwrDpjW?<rrNmr$fl{nAbGPLGa5u>UgQ0k`7#pid_1d-0;WCcc0l{#{ve*tBHYHtg1kP~~PK4JuP4uT#))dQRLb(>gZCQC*4-<!~3NS1nVxiF#6_K;WT04HyN6dpxMZ?XJxFPq^jAJ;F5gnO=m6@65oaqa(vXV8{hRCVpYBilC&*idm#^M@5Q!KJE0*g%abu?B8Q)SYhPKU(}c_joC4LrY!os8CrX{*~Isj_kCH8h5l0F_+A?OKuy`a*7HeZgc6%z_t)r7#4yH@Hc`avzW3-r1>g<<10SDsC4MX)8$WD)f17P^sR_-(q2Za!9s7*EHIupbRgZTOIlJV`fyn>KyB7cJ{iJiai~BgSfg(S)6I2~c-4(|!XvM)Auk`mhySHHo04c=HW>5eUK;6aGA<ySPpktAs(Q?|;?;dykJrCy8Zh3v85vO@RK<a#%vfiexmC>tt^^-Nef(;j`#PfgFe<?Uh)(rK^k!txt;HhmF*=@%wpe192R`76cBD%~)gwNMCJx_?XTn~vi<)OF+2~U#rh2PPrtv6aWZF_;DIL^EwxfYPxPERzc@fgm{jr{MI#gZtkL<G1>c?ja5*<~hz?uF!jzpHjf0I`S3OD=V7%2sg8%t|-`P;Cz~OQ4+eh;el8ENnCjyFKLp`a%-$C9O_v$np4(qokqa2QFGCb`W8VC+l?t%9Cr(Kp2f5tD7lxZ$4R`>Kug8p#xCgE}8LlujkHcC8_kkeUpHkv&9fX6h!2cdNJFFjp~;fSGkg>ik5Z#$&*_0#TPgUS`Im*ra4=ICt9`PWy;G=z9YC5*}8y?^e#ROCCvg{j$ifgP{eE2YAnfuk7mGVjfmA1l_N&!1#ZxPM!}G3ZeDf`b=V&3kh;FuL$A(2Dx9@*t`di3Y>+oH1ZhC4>k!WxjP@jcWB0clD&vhKzMuje5<Cz90000X!3!g@JKfz`>BAa1LD2m5m7Kr}P>gHZ5F!%HI4aE)^gt|X*iU7fQn)5IGqAgPBH;sr{f`nxVg{`1pwDZE(GIjb;ScGEEpha1n|oS|nAveG`~;V488tuv02cXjfB*mh15P0(xiJnQ@c<j4Fya6J000'

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add New Camera")
        layout = QVBoxLayout()
        self.setLayout(layout)

        form_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("ASI2400MC Pro")
        self.name_input.setToolTip("The name of the camera")
        form_layout.addRow("Camera Name:", self.name_input)

        self.qe_red_ha = QLineEdit()
        self.qe_red_ha.setPlaceholderText("0.8")
        self.qe_red_ha.setToolTip("The quantum efficiency of the red channel at the hydrogen-alpha wavelength")
        form_layout.addRow("Red QE at Hα:", self.qe_red_ha)

        self.qe_red_oiii = QLineEdit()
        self.qe_red_oiii.setPlaceholderText("0.03")
        self.qe_red_oiii.setToolTip("The quantum efficiency of the red channel at the oxygen-III wavelength")
        form_layout.addRow("Red QE at OIII:", self.qe_red_oiii)

        self.qe_green_ha = QLineEdit()
        self.qe_green_ha.setPlaceholderText("0.15")
        self.qe_green_ha.setToolTip("The quantum efficiency of the green channel at the hydrogen-alpha wavelength")
        form_layout.addRow("Green QE at Hα:", self.qe_green_ha)

        self.qe_green_oiii = QLineEdit()
        self.qe_green_oiii.setPlaceholderText("0.92")
        self.qe_green_oiii.setToolTip("The quantum efficiency of the green channel at the oxygen-III wavelength")
        form_layout.addRow("Green QE at OIII:", self.qe_green_oiii)

        self.qe_blue_ha = QLineEdit()
        self.qe_blue_ha.setPlaceholderText("0.04")
        self.qe_blue_ha.setToolTip("The quantum efficiency of the blue channel at the hydrogen-alpha wavelength")
        form_layout.addRow("Blue QE at Hα:", self.qe_blue_ha)

        self.qe_blue_oiii = QLineEdit()
        self.qe_blue_oiii.setPlaceholderText("0.5")
        self.qe_blue_oiii.setToolTip("The quantum efficiency of the blue channel at the oxygen-III wavelength")
        form_layout.addRow("Blue QE at OIII:", self.qe_blue_oiii)

        layout.addLayout(form_layout)
        layout.addStretch()

        from base64 import b85decode
        image = QImage.fromData(b85decode(self.EXAMPLE_IMAGE), format="webp")

        explanation = QLabel(
            "You can obtain the quantum efficiency data for your camera from graphs "
            "provided by the manufacturer, like this one for the ZWO ASI2400MC Pro:"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        image_label = QLabel()
        image_label.setPixmap(QPixmap.fromImage(image).scaledToWidth(400, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_camera)
        layout.addWidget(add_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        layout.addWidget(cancel_button)

    def add_camera(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.critical(self, "Error", "Camera name cannot be empty.")
            return
        try:
            cameras[name] = {
                "qe_r": {
                    "ha": float(self.qe_red_ha.text().strip()),
                    "oiii": float(self.qe_red_oiii.text().strip()),
                },
                "qe_g": {
                    "ha": float(self.qe_green_ha.text().strip()),
                    "oiii": float(self.qe_green_oiii.text().strip()),
                },
                "qe_b": {
                    "ha": float(self.qe_blue_ha.text().strip()),
                    "oiii": float(self.qe_blue_oiii.text().strip()),
                },
            }
        except ValueError:
            QMessageBox.critical(self, "Error", "Quantum efficiency values must be valid numbers.")
            return
        with open(CAMERAS_FILE, "w") as f:
            json.dump(cameras, f, indent=4)
        self.close()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Duosplit")

        main_layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        self.dropdown = QComboBox()
        self.dropdown.activated.connect(self.handle_dropdown)
        form_layout.addRow("Camera:", self.dropdown)
        self.refresh_dropdown()

        self.population_input = QLineEdit()
        self.population_input.setText("100")
        self.population_input.setToolTip("Population size for the genetic algorithm")
        self.generations_input = QLineEdit()
        self.generations_input.setText("250")
        self.generations_input.setToolTip("Number of generations for the genetic algorithm")
        self.elitism_input = QLineEdit()
        self.elitism_input.setText("5")
        self.elitism_input.setToolTip("Number of elite individuals to carry over each generation")
        self.initial_std_input = QLineEdit()
        self.initial_std_input.setText("0.5")
        self.initial_std_input.setToolTip("Initial standard deviation for mutation")
        self.decay_rate_input = QLineEdit()
        self.decay_rate_input.setText("0.1")
        self.decay_rate_input.setToolTip("Decay rate for mutation standard deviation")

        form_layout.addRow("Population:", self.population_input)
        form_layout.addRow("Generations:", self.generations_input)
        form_layout.addRow("Elitism:", self.elitism_input)
        form_layout.addRow("Initial Standard Deviation:", self.initial_std_input)
        form_layout.addRow("Decay Rate:", self.decay_rate_input)

        main_layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        run_button = QPushButton("Run")
        cancel_button = QPushButton("Cancel")

        run_button.clicked.connect(self.run_action)
        cancel_button.clicked.connect(self.close)

        button_layout.addWidget(run_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

    def handle_dropdown(self, index):
        if index == self.dropdown.count() - 1:  # last item is "Add new..."
            dialog = NewCameraDialog()
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.show()
            loop = QEventLoop()
            dialog.destroyed.connect(loop.quit)
            loop.exec()
            self.refresh_dropdown()
            self.dropdown.setCurrentIndex(0)

    def refresh_dropdown(self):
        self.dropdown.clear()
        for camera in cameras.keys():
            self.dropdown.addItem(camera)
        self.dropdown.addItem("Add new...")

    def run_action(self):
        try:
            camera = cameras[self.dropdown.currentText()]
            parameters = Parameters(
                qe_r_ha=camera["qe_r"]["ha"],
                qe_r_oiii=camera["qe_r"]["oiii"],
                qe_g_ha=camera["qe_g"]["ha"],
                qe_g_oiii=camera["qe_g"]["oiii"],
                qe_b_ha=camera["qe_b"]["ha"],
                qe_b_oiii=camera["qe_b"]["oiii"],
                population=int(self.population_input.text().strip()),
                generations=int(self.generations_input.text().strip()),
                elitism=int(self.elitism_input.text().strip()),
                initial_std_dev=float(self.initial_std_input.text().strip()),
                decay_rate=float(self.decay_rate_input.text().strip()),
            )
        except ValueError:
            QMessageBox.critical(self, "Error", "Please ensure all parameters are valid numbers.")
            return
        self.close()
        run_duosplit(parameters)


app = QApplication([])
window = MainWindow()
window.resize(400, 400)
window.show()
app.exec()
