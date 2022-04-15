This repo contains the code used for experiments from paper "Multilingual fine-tuning for Grammatical Error Correction" (https://doi.org/10.1016/j.eswa.2022.116948).

To run the scripts you need to acquire the following data sets):

 * ARAB (ar): https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/
 * CHECH (ch): https://github.com/adrianeboyd/boyd-wnut2018#download-data
 * GERMAN (de): https://github.com/adrianeboyd/boyd-wnut2018#download-data
 * RUSSIAN (ru): https://github.com/arozovskaya/RULEC-GEC
 * ROMANIAN (ro): https://github.com/teodor-cotet/RoGEC (only RONACC https://nextcloud.readerbench.com/index.php/s/9pwymesT5sycxoM)
 * ENGLISH (en): 
	** W&I+LOCNESS v2.1: https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
	** FCE v2.1: https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz
 * CHINESEE (ch): http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata02.tar.gz


Data for respective languages should be stored in separate folders named after language code (so as in bracelets above: ar, ch, de, ru, ro, en, ch). 

Every folder should contain files named:
 * train.source
 * train.target
 * val.source
 * val.target

and for evaluation
 * {lang_code}-{dev,test}.src
 * {lang_code}-{dev,test}.trg

so, for example
 * ar-dev.src
 * ar-dev.trg
 * ar-test.src
 * ar-test.trg


To set up required python dependencies you should run prepare-env.sh (it was developed on Python 3.7.6, worked on 3.8.12 as well). 