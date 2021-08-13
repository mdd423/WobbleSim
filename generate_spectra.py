import argparse
import simulator
import numpy as np
import matplotlib.pyplot as plt

import h5py


def plot_epoch(epoch_idx,lamb,f,ferr,lamb_theory,f_star=None,f_tell=None,f_gas=None):
    plt.figure(figsize=(20,8))
    plt.title('wobble toy data')
    plt.xlabel('$\lambda_{%i}$' % epoch_idx)
    plt.ylabel('$f_{%i}$' % epoch_idx)
    # plt.xlim(15000,16000)
    # plt.ylim(0,1.2)
    if f_star is not None:
        plt.plot(lamb_theory,f_star[epoch_idx,:],'red',alpha=0.5,label='star')
    if f_tell is not None:
        plt.plot(lamb_theory,f_tell[epoch_idx,:],'blue',alpha=0.5,label='telluric')
    if f_gas is not None:
        plt.plot(lamb_theory,f_gas[epoch_idx,:],'green',alpha=0.5,label='gas cell')
    plt.errorbar(lamb,f[epoch_idx,:],ferr[epoch_idx,:],fmt='.k',elinewidth=0.7,zorder=1,alpha=0.4,ms=6,label='data')
    plt.legend()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',action='store',default=20000,type=np.float32)
    parser.add_argument('--vp',action='store',default=30,type=np.float32)
    parser.add_argument('--w',action='store',default=1.0,type=np.float32)
    parser.add_argument('--epsilon',action='store',default=0.0001,type=np.float32)
    parser.add_argument('--gamma',action='store',default=1.0,type=np.float32)
    parser.add_argument('--s2n',action='store',default=20,type=np.float32)
    parser.add_argument('--epoches',action='store',default=5,type=int)

    parser.add_argument('--stellarname_wave',action='store',default='data/stellar/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits',type=str)
    parser.add_argument('--stellarname_flux',action='store',default='data/stellar/PHOENIX/lte02400-0.50-4.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',type=str)
    parser.add_argument('--skycalcname',action='store',default='data/tellurics/skycalc/skycalc_defaults.txt',type=str)
    parser.add_argument('--skycalcalma',action='store',default='data/tellurics/skycalc/almanac_example.txt',type=str)
    parser.add_argument('--gascellname',action='store',default='data/gascell/keck_fts_renorm.idl',type=str)
    args   = parser.parse_args()


    out_name = "out/sim_ful_lr{}_s{}_e{}_v{}_ep{}_g{}_w{}.h5".format(args.lr
                        ,args.s2n,args.epoches
                        ,args.vp,args.epsilon,args.gamma,args.w)
    out = simulator.main(args.lr
                        ,args.s2n,args.epoches
                        ,args.vp,args.epsilon,args.gamma,args.w
                        ,args.stellarname_flux,args.stellarname_flux
                        ,args.skycalcname,args.skycalcalma
                        ,args.gascellname)

    hf = h5py.File(out_name,"w")
    theory_group = hf.create_group("theory")
    theory_group.create_dataset("wavelength",data=out["wavelength_theory"])
    theory_group.create_dataset("flux_stellar",data=out["flux_stellar"])
    theory_group.create_dataset("flux_tellurics",data=out["flux_tellurics"])
    theory_group.create_dataset("flux_gas",data=out["flux_gas"])

    sample_group = hf.create_group("samples")
    sample_group.create_dataset("wavelength",data=out["wavelength_sample"])
    sample_group.create_dataset("flux",data=out["flux"])
    sample_group.create_dataset("flux_error",data=out["flux_error"])

    consts_group = hf.create_group("constants")
    consts_group.create_dataset("m",data=out["m"])
    consts_group.create_dataset("del",data=out["del"])
    consts_group.create_dataset("airmass",data=out["airmass"])
    consts_group.create_dataset("delta",data=out["delta"])
    consts_group.create_dataset("low_resolution",data=args.lr)

    hf.close()

    # df.to_csv(out_name)
    # Plot an epoch
    ####################################################
    # epoch_idx = 17
    # plot_epoch(epoch_idx,lamb,f,ferr,xs,y_star,y_tell,y_gas)
    # plt.savefig('out/wobble_toy_e{}.png'.format(epoch_idx))
    # plt.show()
