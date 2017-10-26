import pandas as pd

df = pd.read_csv('./car-eval.csv')

debug_log_enabled = False
priorName = 'class'
totalRows = len(df)
priors = df[priorName].unique()
priorDict = dict()
for prior in priors:
    priorDict[prior] = len(df[(df[priorName] == prior)]) / totalRows


class EvidenceAndLikelyHood(object):
    def __init__(self, featureName):
        features = df[featureName].unique()
        self.evidenceDict = dict()
        for feature in features:
            self.evidenceDict[feature] = len(df[df[featureName] == feature]) / totalRows
        self.featureLikelyhood = dict()
        for feature in features:
            for prior in priors:
                tuple = (feature, prior)
                tmpDf = df[df[priorName] == prior]
                tmpLen = len(tmpDf)
                frequency = len(tmpDf[tmpDf[featureName] == feature]) / tmpLen
                self.featureLikelyhood[tuple] = frequency


buyingEvidenceAndLikelyhood = EvidenceAndLikelyHood('buying')
maintEvidenceAndLikelyhood = EvidenceAndLikelyHood('maint')
doorsEvidenceAndLikelyhood = EvidenceAndLikelyHood('doors')
personsEvidenceAndLikelyhood = EvidenceAndLikelyHood('persons')
lug_boot_evidence_and_likelyhood = EvidenceAndLikelyHood('lug_boot')
safety_evidence_and_likelyhood = EvidenceAndLikelyHood('safety')


def calculate_posterior(pr, **kwargs):
    nominator = []
    nominator.append(priorDict[pr])
    denominator = []

    buying = kwargs.get('buying', None)
    maint = kwargs.get('maint', None)
    doors = kwargs.get('doors', None)
    persons = kwargs.get('persons', None)
    lug_boot = kwargs.get('lug_boot', None)
    safety = kwargs.get('safety', None)
    if buying is not None:
        nominator.append(buyingEvidenceAndLikelyhood.featureLikelyhood[(buying, pr)])
        denominator.append(buyingEvidenceAndLikelyhood.evidenceDict[buying])
    if maint is not None:
        nominator.append(maintEvidenceAndLikelyhood.featureLikelyhood[(maint, pr)])
        denominator.append(maintEvidenceAndLikelyhood.evidenceDict[maint])
    if doors is not None:
        nominator.append(doorsEvidenceAndLikelyhood.featureLikelyhood[(doors, pr)])
        denominator.append(doorsEvidenceAndLikelyhood.evidenceDict[doors])
    if persons is not None:
        nominator.append(personsEvidenceAndLikelyhood.featureLikelyhood[(persons, pr)])
        denominator.append(personsEvidenceAndLikelyhood.evidenceDict[persons])
    if lug_boot is not None:
        nominator.append(lug_boot_evidence_and_likelyhood.featureLikelyhood[(lug_boot, pr)])
        denominator.append(lug_boot_evidence_and_likelyhood.evidenceDict[lug_boot])
    if safety is not None:
        nominator.append(safety_evidence_and_likelyhood.featureLikelyhood[(safety, pr)])
        denominator.append(safety_evidence_and_likelyhood.evidenceDict[safety])

    nominator_sum = 1
    for nom_val in nominator:
        nominator_sum *= nom_val
    denominator_sum = 1
    for denom_val in denominator:
        denominator_sum *= denom_val
    return nominator_sum / denominator_sum


def predict_class(**kwargs):
    prior_probabilities = []
    prior_propbability_to_prior = dict()
    for pr in priors:
        posterior = calculate_posterior(pr, **kwargs)
        prior_propbability_to_prior[posterior] = pr
        prior_probabilities.append(posterior)

    prior_probabilities.sort(reverse=True)
    predicted_class = prior_propbability_to_prior[prior_probabilities[0]]
    predicted_class_probability = prior_probabilities[0]
    if debug_log_enabled is True:
        print('Predicted ', predicted_class, ' with ', predicted_class_probability, ' probability.')
    return prior_propbability_to_prior[prior_probabilities[0]]


correct_predictions = 0
for index,row in df.iterrows():
    buying = row['buying']
    maint = row['maint']
    doors = row['doors']
    persons = row['persons']
    lug_boot = row['lug_boot']
    safety = row['safety']
    classN = row['class']
    predicted_class = predict_class(buying=buying,maint=maint,doors=doors,persons=persons,lug_boot=lug_boot,safety=safety)
    if classN == predicted_class:
        correct_predictions += 1

print('Total trials:', totalRows, ', correct predictions:', correct_predictions, ', wrong predictions:', totalRows-correct_predictions, ', accuracy:', correct_predictions*100/totalRows, '%')

