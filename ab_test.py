import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(42)

n_control = 2000
n_treatment = 2000
true_control_rate = 0.10
true_treatment_rate = 0.125

control_conversions = np.random.binomial(1, true_control_rate, n_control)
treatment_conversions = np.random.binomial(1, true_treatment_rate, n_treatment)

print(f"Control:   {control_conversions.sum()} conversions / {n_control} visitors ({control_conversions.mean():.2%})")
print(f"Treatment: {treatment_conversions.sum()} conversions / {n_treatment} visitors ({treatment_conversions.mean():.2%})")

if __name__ == '__main__':
    with pm.Model() as ab_model:
        p_control = pm.Beta('p_control', alpha=1, beta=1)
        p_treatment = pm.Beta('p_treatment', alpha=1, beta=1)

        obs_control = pm.Binomial('obs_control', n=n_control, p=p_control,
                                   observed=control_conversions.sum())
        obs_treatment = pm.Binomial('obs_treatment', n=n_treatment, p=p_treatment,
                                     observed=treatment_conversions.sum())

        relative_uplift = pm.Deterministic('relative_uplift',
                                            (p_treatment - p_control) / p_control)

        trace = pm.sample(2000, tune=1000, chains=1, cores=1,
                          return_inferencedata=True,
                          progressbar=True, random_seed=42)

    prob_treatment_better = (trace.posterior['p_treatment'].values >
                              trace.posterior['p_control'].values).mean()

    print(f"\nP(treatment > control): {prob_treatment_better:.2%}")
    print(f"Expected uplift: {trace.posterior['relative_uplift'].values.mean():.2%}")
    print(az.summary(trace, var_names=['p_control', 'p_treatment', 'relative_uplift']))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(trace.posterior['p_control'].values.flatten(), bins=50,
                 alpha=0.6, color='steelblue', label='Control')
    axes[0].hist(trace.posterior['p_treatment'].values.flatten(), bins=50,
                 alpha=0.6, color='coral', label='Treatment')
    axes[0].set_xlabel('Conversion Rate')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Posterior Distributions: Conversion Rate')
    axes[0].legend()

    uplift_samples = trace.posterior['relative_uplift'].values.flatten()
    axes[1].hist(uplift_samples, bins=50, alpha=0.7, color='mediumseagreen')
    axes[1].axvline(0, color='red', linestyle='--', label='No uplift')
    axes[1].axvline(uplift_samples.mean(), color='black',
                    linestyle='--', label=f'Mean: {uplift_samples.mean():.1%}')
    axes[1].set_xlabel('Relative Uplift')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Posterior Distribution: Relative Uplift')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('posterior_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    avg_order_value = 45.00
    monthly_visitors = 50000

    control_rate_samples = trace.posterior['p_control'].values.flatten()
    treatment_rate_samples = trace.posterior['p_treatment'].values.flatten()

    revenue_control = control_rate_samples * monthly_visitors * avg_order_value
    revenue_treatment = treatment_rate_samples * monthly_visitors * avg_order_value
    revenue_lift = revenue_treatment - revenue_control

    print(f"\n--- Expected Monthly Revenue Impact ---")
    print(f"Control revenue:    £{revenue_control.mean():,.0f}")
    print(f"Treatment revenue:  £{revenue_treatment.mean():,.0f}")
    print(f"Expected lift:      £{revenue_lift.mean():,.0f} per month")
    print(f"95% credible interval: £{np.percentile(revenue_lift, 2.5):,.0f} to £{np.percentile(revenue_lift, 97.5):,.0f}")
    print(f"P(positive revenue impact): {(revenue_lift > 0).mean():.2%}")