# Business case

## The decision

An operator of a turbofan fleet must decide, for each engine and each cycle, **whether to pull it for maintenance
now or let it keep flying**. The model improves this scheduling decision by estimating how close each engine is to
failure.

**Who decides:** maintenance planning, with engineering sign-off. The model supplies a recommendation and an
estimate of risk; it does not remove the engine.

**Constraint:** a shop visit cannot be arranged instantly. It needs a lead time of N cycles, set to 30 in
`business.maintenance_lead_time_cycles` (ASSUMPTION). The useful question is therefore "will this engine fail
within the next N cycles?", which is why the classification target uses the same N.

## Baselines (what happens without a model)

1. **Run to failure:** maintain only after failure. No useful life is wasted, but every removal is unplanned.
2. **Fixed-interval maintenance:** pull every engine after a fixed number of cycles, set from fleet history. It
   avoids most failures but pulls many healthy engines early.

Phase 6 compares the model-driven policy with both.

## Cost of each error (ASSUMPTIONS, `business` in `configs/config.yaml`)

| Outcome | What happens | Cost used |
|---|---|---|
| **Late (missed failure)** | In-service failure: unscheduled removal, disruption, possible secondary damage and a safety event | `unplanned_failure_cost` = 1,000 (thousand currency units) |
| **Planned maintenance** | Scheduled shop visit at the chosen cycle | `planned_maintenance_cost` = 200 |
| **Early (lost useful life)** | Each cycle of remaining life given up by pulling early | `lost_life_cost_per_cycle` = 1.0 per cycle |

- **Asymmetry:** errors are deliberately asymmetric, with a late pull far costlier than an early one. The same
  asymmetry sits in the NASA scoring function, which penalises late RUL predictions more than early ones.
- **Assumptions only:** no real cost data accompany C-MAPSS. All four values are assumptions, varied in phase 6,
  and every cost figure is reported as an estimate under them.

## What the data can and cannot support

C-MAPSS is a **simulation** (NASA's Commercial Modular Aero-Propulsion System Simulation) with injected degradation
and sensor noise. It is the standard public benchmark for remaining-useful-life prediction, which makes results
comparable with the literature, but:
- the engines, fault modes and noise are simulated, so no claim here transfers to a real fleet without validation
  on real data;
- each engine's run has a clean, single degradation path to failure, which is cleaner than real maintenance
  records with repairs, sensor faults and censoring;
- costs and lead times are assumptions, not airline data.

The deliverable is therefore a working, tested decision pipeline and a quantified policy comparison *under stated
assumptions on simulated data*, not a validated fleet saving.
