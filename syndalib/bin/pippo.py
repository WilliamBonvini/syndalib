from syndalib import syn2d


def run():
    syn2d.generate_data(ns=16,
                        npps=256,
                        class_type="circles",
                        nm=2,
                        outliers_range=[0.10],
                        noise_range=[0.01],
                        ds_name="no_boh_noise_variant",
                        plot=True,
                        is_train=False,
                        save_matlab=False)




if __name__ == "__main__":
    run()
