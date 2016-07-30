#!/usr/bin/python
criterion_l = ['gini','entropy'] #2
max_depth_l = [None, 2,5,10,20,50,100] #7 
min_samples_split_l = [1,2,3,4,5] #5 
min_samples_leaf_l = [1,2,3,4,5] #5 
criterion_a = []
max_depth_a = []
min_samples_split_a = []
min_samples_leaf_a =[]
recall_a =[]
precision_a=[]
f1_a=[]
import csv
f = open('outputs.csv', 'wb')
writer = csv.writer(f)
writer.writerow(["criterion","max_depth","min_samples_split","min_samples_leaf","recall","precision","f1"])

for criterion in criterion_l:
	for max_depth in max_depth_l:
		for min_samples_split in min_samples_split_l:
			for min_samples_leaf in min_samples_leaf_l:
				import numpy as np
				from sklearn.metrics import accuracy_score
				from time import time
				import sys
				import pickle
				sys.path.append("../tools/")

				from feature_format import featureFormat, targetFeatureSplit
				from tester import dump_classifier_and_data

				### Task 1: Select what features you'll use.
				### features_list is a list of strings, each of which is a feature name.
				### The first feature must be "poi".
				features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

				### Load the dictionary containing the dataset
				with open("final_project_dataset.pkl", "r") as data_file:
					data_dict = pickle.load(data_file)

				#print "\n---DATA EXPLORATION---"
				#print "Total Number of Data Points: " + str(len(data_dict))

				poi_count = 0
				for key, value in data_dict.items():
					if data_dict[key]["poi"]:
						poi_count+=1
				#print "POI: " + str(poi_count), "Non-POI: " + str(len(data_dict)-poi_count)

				total_missings = 0
				for j in range(len(features_list)-1):
					missings = 0
					for key, value in data_dict.items():
						if data_dict[key][features_list[j+1]]=="NaN":
							missings +=1
					total_missings += missings
				#	print str(missings)+" missing values in "+str(features_list[j+1])
				#print str(total_missings)+" total missing values in dataset for selected "+str(len(features_list)-1)+" features"
					
				#print "\n" + str(len(features_list)-1) + " features are;"
				sd_list = []
				mean_list = []
				for j in range(len(features_list)-1):
				#	print features_list[j+1]
					temp_list = []
					for key, value in data_dict.items():
						if data_dict[key][features_list[j+1]]=="NaN":
							data_dict[key][features_list[j+1]]=0
						temp_list.append(float(data_dict[key][features_list[j+1]]))
				#	print "max: " + str(max(temp_list))
				#	print "min: " + str(min(temp_list))
				#	print "range: " + str(max(temp_list)-min(temp_list))
					sd_list.append(np.std(temp_list))
				#	print "SD: " + str(np.std(temp_list))
					mean_list.append(np.mean(temp_list))
				#	print "mean: " + str(np.mean(temp_list)) + "\n"		

				### Task 2: Remove outliers
				#print "\n---ROMOVING OUTLIERS---"
				sd_limit = 3 #data points to be kept in this sd range 
				removal = 0
				data_dict.pop("TOTAL", None)
				#print str(removal)+" outliers removed"
				#print "Total number of data points after removing outliers: " + str(len(data_dict))

				### Task 3: Create new feature(s)
				#print "\n---NEW FEATURES---"
				for key, val in data_dict.items():
					if data_dict[key]['to_messages'] != 0:
						data_dict[key]['to_poi_ratio'] = data_dict[key]['from_poi_to_this_person']/float(data_dict[key]['to_messages'])
					else:
						data_dict[key]['to_poi_ratio'] = 0.0
					if data_dict[key]['from_messages'] != 0:
						data_dict[key]['from_poi_ratio'] = data_dict[key][ 'from_this_person_to_poi']/float(data_dict[key]['from_messages'])
					else:
						data_dict[key]['from_poi_ratio'] = 0.0

				features_list.append('to_poi_ratio')
				features_list.append('from_poi_ratio')
				#print "New features 'to_poi_ratio' and 'from_poi_ratio' are created"

				### Store to my_dataset for easy export below.
				my_dataset = data_dict

				### Extract features and labels from dataset for local testing
				data = featureFormat(my_dataset, features_list, sort_keys = True)
				labels, features = targetFeatureSplit(data)

				#print "\n---FEATURE SELECTION---"
				from sklearn.feature_selection import SelectKBest, f_classif
				k = 3 #best features to be chosen 3
				#print "First values of features", features[0].shape, features[0]
				selector = SelectKBest(f_classif, k=k)
				features = selector.fit_transform(features, labels)
				#print "First values of best {} features".format(k), features.shape, features[0]
				chosen = np.asarray(features_list[1:])[selector.get_support()]
				#print selector.scores_
				scores = selector.scores_
				index = np.argsort(scores)[::-1]
				#print 'Feature Ranking: '
				for i in range(len(features_list)-1):
					print "{}. feature {} ({})".format(i+1,features_list[index[i]+1],scores[index[i]])
				print "Chosen ones: ", chosen
				### Task 4: Try a varity of classifiers
				### Please name your classifier clf for easy export below.
				### Note that if you want to do PCA or other multi-stage operations,
				### you'll need to use Pipelines. For more info:
				### http://scikit-learn.org/stable/modules/pipeline.html

				# Provided to give you a starting point. Try a variety of classifiers.
				from sklearn.grid_search import GridSearchCV
				from sklearn.naive_bayes import GaussianNB
				from sklearn.svm import SVC
				from sklearn import tree
				from sklearn.ensemble import RandomForestClassifier
				from sklearn.metrics import make_scorer, f1_score
				#f1_scorer = make_scorer(f1_score, pos_label=1)
				#param_random = {'n_estimators': [1,2,3,5,10,100], 'max_features': range(1,k+1),'criterion': ['gini', 'entropy']} #bests are 1 for both
				#param_random = {'random_state': range(0,100)}
				#clf = GridSearchCV(RandomForestClassifier(random_state=4), param_random, scoring=f1_scorer)
				#clf = RandomForestClassifier(n_estimators=1, max_features=2, random_state=4)
				#param_svc = {'C': [1, 10, 100, 1e3, 1e4, 1e5], 'degree': [2,3,4],'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
				#clf = GridSearchCV(SVC(kernel='rbf',random_state=42), param_svc, scoring=f1_scorer)
				#clf = SVC(kernel='poly', degree=4, C=1000) #because GridSearchCV(SVC(kernel='poly')) doesn't work
				#clf = RandomForestClassifier()
				#clf = SVC()
				#clf = GaussianNB()

				clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

				from sklearn import decomposition
				from sklearn.pipeline import Pipeline
				from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer
				scaler = MinMaxScaler()
				pca = decomposition.PCA()
				classifier = tree.DecisionTreeClassifier()
				steps = [('features', scaler), ('pca', pca),('classify', classifier)]
				pipe = Pipeline(steps)
				#clf = GridSearchCV(pipe, dict(pca__n_components=[None,2,3], classify__min_samples_split=[2,6,10], 
				#							  classify__criterion=['gini','entropy']),
				#							  scoring='f1')
				#clf = Pipeline([('features', scaler), ('pca', decomposition.PCA()),('classify', tree.DecisionTreeClassifier())]) #THE BEST
				### Task 5: Tune your classifier to achieve better than .3 precision and recall 
				### using our testing script. Check the tester.py script in the final project
				### folder for details on the evaluation method, especially the test_classifier
				### function. Because of the small size of the dataset, the script uses
				### stratified shuffle split cross validation. For more info: 
				### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

				# Example starting point. Try investigating other evaluation techniques!
				print "\n---CLASSIFIER---"
				from sklearn.cross_validation import train_test_split
				features_train, features_test, labels_train, labels_test = \
					train_test_split(features, labels, test_size=0.3, random_state=42)

				t0 = time()
				reg = clf.fit(features_train, labels_train)
				print "training time:", round(time()-t0, 3), "s"
				try:
					best_params = clf.best_params_
					print "Best parameters: ", best_params
				except:
					pass

				t1 = time()
				pred = clf.predict(features_test)
				print "predict time:", round(time()-t1, 3), "s"

				accuracy = accuracy_score(labels_test, pred)
				print "accuracy: " +  str(accuracy)
				from sklearn.metrics import classification_report
				print classification_report(labels_test, pred)
				#print f1_score(labels_test, pred)

				try:
					print "\n---FEATURE IMPORTANCES---"
					try: #GridSearchCV + Pipeline
						if best_params['pca__n_components'] != None:
							n = best_params['pca__n_components'] #chosen PCA components
						else:
							n = k
						fit_pipeline = clf.best_estimator_
						fit_clf = fit_pipeline.steps[-1][1]
					except: #Just Pipeline
						fit_clf = clf.steps[-1][1]
						n = k

					importances = fit_clf.feature_importances_
					indices = np.argsort(importances)[::-1]
					print 'Feature Ranking: '
					for i in range(n):
						print "{}. component ({})".format(i+1,importances[indices[i]])
				except:
					pass

				### Task 6: Dump your classifier, dataset, and features_list so anyone can
				### check your results. You do not need to change anything below, but make sure
				### that the version of poi_id.py that you submit can be run on its own and
				### generates the necessary .pkl files for validating your results.
				chosen_list = chosen.tolist()
				chosen_list.insert(0, "poi")
				features_list = chosen_list

				dump_classifier_and_data(clf, my_dataset, features_list)
				import time
				time.sleep(0.5)

				import pickle
				import sys
				from sklearn.cross_validation import StratifiedShuffleSplit
				sys.path.append("../tools/")
				from feature_format import featureFormat, targetFeatureSplit

				PERF_FORMAT_STRING = "\
				\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
				Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
				RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
				\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


				CLF_PICKLE_FILENAME = "my_classifier.pkl"
				DATASET_PICKLE_FILENAME = "my_dataset.pkl"
				FEATURE_LIST_FILENAME = "my_feature_list.pkl"

				def dump_classifier_and_data(clf, dataset, feature_list):
					with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
						pickle.dump(clf, clf_outfile)
					with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
						pickle.dump(dataset, dataset_outfile)
					with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
						pickle.dump(feature_list, featurelist_outfile)

				def load_classifier_and_data():
					with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
						clf = pickle.load(clf_infile)
					with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
						dataset = pickle.load(dataset_infile)
					with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
						feature_list = pickle.load(featurelist_infile)
					return clf, dataset, feature_list


				### load up student's classifier, dataset, and feature_list
				clf, dataset, feature_list = load_classifier_and_data()
				### Run testing script
				data = featureFormat(dataset, feature_list, sort_keys = True)
				labels, features = targetFeatureSplit(data)
				cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
				true_negatives = 0
				false_negatives = 0
				true_positives = 0
				false_positives = 0
				for train_idx, test_idx in cv: 
					features_train = []
					features_test  = []
					labels_train   = []
					labels_test    = []
					for ii in train_idx:
						features_train.append( features[ii] )
						labels_train.append( labels[ii] )
					for jj in test_idx:
						features_test.append( features[jj] )
						labels_test.append( labels[jj] )
					
					### fit the classifier using training set, and test on test set
					clf.fit(features_train, labels_train)
					predictions = clf.predict(features_test)
					for prediction, truth in zip(predictions, labels_test):
						if prediction == 0 and truth == 0:
							true_negatives += 1
						elif prediction == 0 and truth == 1:
							false_negatives += 1
						elif prediction == 1 and truth == 0:
							false_positives += 1
						elif prediction == 1 and truth == 1:
							true_positives += 1
						else:
							print "Warning: Found a predicted label not == 0 or 1."
							print "All predictions should take value 0 or 1."
							print "Evaluating performance for processed predictions:"
							break
				try:
					total_predictions = true_negatives + false_negatives + false_positives + true_positives
					accuracy = 1.0*(true_positives + true_negatives)/total_predictions
					precision = 1.0*true_positives/(true_positives+false_positives)
					recall = 1.0*true_positives/(true_positives+false_negatives)
					f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
					f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
					print clf
					print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
					print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
					print ""
				except:
					print "Got a divide by zero when trying out:", clf
					print "Precision or recall may be undefined due to a lack of true positive predicitons."

					
				criterion_a.append(criterion)
				max_depth_a.append(max_depth)
				min_samples_split_a.append(min_samples_split)
				min_samples_leaf_a.append(min_samples_leaf)
				recall_a.append(recall)
				precision_a.append(precision)
				f1_a.append(f1)
				writer.writerow([criterion,max_depth,min_samples_split,min_samples_leaf,recall,precision,f1])
				
				
print criterion_a, f1_a