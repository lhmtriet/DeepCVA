#!/bin/bash

## Evaluate different sampling strategies
for sampling in 'none over over+'; do
	## Validate - No Dependencies
	while read a b c d e f g; do
	    python3 ml_model/model_evaluation.py "${a}" "${b}" "${c}" "${d}" "${e}" "${f}" "${g}" "${sampling}"
	done < <(sed 's/,/\t/g' ml_model/cvss_inputs.csv)
	## Test
	while read a b c d e f g; do
	    python3 ml_model/model_evaluation.py "${a}" "${b}" "${c}" "${d}" "${e}" "${f}" "${g}" "${sampling}"
	done < <(sed 's/,/\t/g' ml_model/cvss_inputs.csv)
done

## Average Folds
python3 ml_model/model_evaluation.py average cvss2_vector 0 - - - - -
python3 ml_model/model_evaluation.py average cvss2_accessvect 0 - - - - -
python3 ml_model/model_evaluation.py average cvss2_accesscomp 0 - - - - -
python3 ml_model/model_evaluation.py average cvss2_auth 0 - - - - -
python3 ml_model/model_evaluation.py average cvss2_conf 0 - - - - -
python3 ml_model/model_evaluation.py average cvss2_integrity 0 - - - - -
python3 ml_model/model_evaluation.py average cvss2_avail 0 - - - - -
python3 ml_model/model_evaluation.py average cvss2_severity 0 - - - - -

## Evaluate XCVA
python3 ml_model/extra_evaluate.py
