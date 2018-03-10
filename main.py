from operator import itemgetter
import random
import math
import copy
import itertools
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# GLOBAL VARIABLES

genetic_code = {
	'0000':'0',
	'0001':'1',
	'0010':'2',
	'0011':'3',
	'0100':'4',
	'0101':'5',
	'0110':'6',
	'0111':'7',
	'1000':'8',
	'1001':'9',
	'1010':'+',
	'1011':'-',
	'1100':'*',
	'1101':'/'
	}

solution_found = False
popN = 99 # n number of chromos per population
genesPerCh = 75
max_iterations = 1000
target = 20  # The target number for which we want the expression 
crossover_rate = 0.7
mutation_rate = 0.05
good = []
bad = []
threshold = popN // 5
LOG_DIR = 'C:\\Users\\akash\\Desktop\\tensorboard\\'

"""Generates random population of chromos"""
def generatePop ():
	chromos, chromo = [], []
	for eachChromo in range(popN):
		chromo = []
		for bit in range(genesPerCh * 4):
			chromo.append(random.randint(0,1))
		chromos.append(chromo)
	return chromos

"""Takes a binary list (chromo) and returns a protein (mathematical expression in string)"""
def translate (chromo):
	protein, chromo_string = '',''
	need_int = True
	a, b = 0, 4 # ie from point a to point b (start to stop point in string)
	for bit in chromo:
		chromo_string += str(bit)
	for gene in range(genesPerCh):
		if chromo_string[a:b] == '1111' or chromo_string[a:b] == '1110': 
			continue
		elif chromo_string[a:b] != '1010' and chromo_string[a:b] != '1011' and chromo_string[a:b] != '1100' and chromo_string[a:b] != '1101':
			if need_int == True:
				protein += genetic_code[chromo_string[a:b]]
				need_int = False
				a += 4
				b += 4
				continue
			else:
				a += 4
				b += 4
				continue
		else:
			if need_int == False:
				protein += genetic_code[chromo_string[a:b]]
				need_int = True
				a += 4
				b += 4
				continue
			else:
				a += 4
				b += 4
				continue
	if len(protein) %2 == 0:
		protein = protein[:-1]
	return protein
	
#Evaluation is left to right. No precedence is observed.
#Two operands and an operator are combined.
# Ex: 2+5+7*20/3 will be 7+7*20/3 which will be 14*20/3 = 280/3 = 93.33333
"""Evaluates the mathematical expressions in number + operator blocks of two"""
def evaluate(protein):
	a = 3
	b = 5
	output = -1
	lenprotein = len(protein) 
	if lenprotein == 0:
		output = 0
	if lenprotein == 1:
		output = int(protein)
	if lenprotein >= 3:
		try :
			output = eval(protein[0:3])
		except ZeroDivisionError:
			output = 0
		if lenprotein > 4:
			while b != lenprotein+2:
				try :
					output = eval(str(output)+protein[a:b])
				except ZeroDivisionError:
					output = 0
				a+=2
				b+=2  
	return output

"""Calulates fitness as a fraction of the total fitness"""
def calcFitness (errors):
	fitnessScores = [float(1-i) for i in errors]
	return fitnessScores

def displayFit (error):
	bestFitDisplay = 100
	dashesN = int(error * bestFitDisplay)
	dashes = ''
	for j in range(bestFitDisplay-dashesN):
		dashes+=' '
	for i in range(dashesN):
		dashes+='+'
	return dashes


"""Takes a population of chromosomes and returns a list of tuples where each chromo is paired to its fitness scores and ranked accroding to its fitness"""
def rankPop (chromos):
	proteins, outputs, errors = [], [], []
	i = 1
	# translate each chromo into mathematical expression (protein), evaluate the output of the expression,
	# calculate the inverse error of the output
	print ('%s: %s\t=%s \t%s %s' %('n'.rjust(5), 'PROTEIN'.rjust(30), 'OUTPUT'.rjust(10), 'INVERSE ERROR'.rjust(17), 'GRAPHICAL INVERSE ERROR'.rjust(105)))
	for chromo in chromos: 
		protein = translate(chromo)
		proteins.append(protein)
		
		output = evaluate(protein)
		outputs.append(output)
		
		try:
			error = 1/math.fabs(target-output)
		except ZeroDivisionError:
			global solution_found
			solution_found = True
			error = 0
			print ('\nSOLUTION FOUND' )
			print ('%s: %s \t=%s %s' %(str(i).rjust(5), protein.rjust(30), str(output).rjust(10), displayFit(1.3).rjust(130)))
			break
		else:
			#error = 1/math.fabs(target-output)
			errors.append(error)
		print ('%s: %s \t=%s \t%s %s' %(str(i).rjust(5), protein.rjust(30), str(output).rjust(10), str(error).rjust(17), displayFit(error).rjust(105)))
		i+=1  
	fitnessScores = calcFitness (errors) # calc fitness scores from the erros calculated
	pairedPop = zip ( chromos, proteins, outputs, fitnessScores) # pair each chromo with its protein, output and fitness score
	rankedPop = sorted ( pairedPop,key = itemgetter(-1) ) # sort the paired pop by ascending fitness score
	# print("-----------------------")
	return rankedPop

""" taking a ranked population selects two of the fittest members using roulette method"""
def selectFittest (fitnessScores, rankedChromos):
	while 1 == 1: # ensure that the chromosomes selected for breeding are have different indexes in the population
		index1 = roulette (fitnessScores)
		index2 = roulette (fitnessScores)
		if index1 == index2:
			continue
		else:
			break

	
	ch1 = rankedChromos[index1] # select  and return chromosomes for breeding 
	ch2 = rankedChromos[index2]

	return ch1, ch2

"""Fitness scores are fractions, their sum = 1. Fitter chromosomes have a larger fraction.  """
def roulette (fitnessScores):

	weight_sum = 0.
	scores_len = len(fitnessScores)
	
	for i in range(scores_len):
		weight_sum += fitnessScores[i]

	value = random.uniform(0, 1) * weight_sum	

	for i in range(scores_len):		
		value -= fitnessScores[i]		
		if(value <= 0):
			return i

	return scores_len - 1


def crossover (ch1, ch2):
	r = int(random.uniform(0, 1) * (len(ch1) - 1))
	return ch1[:r]+ch2[r:], ch2[:r]+ch1[r:]


def mutate(ch):
	r = int(random.uniform(0, 1) * (len(ch) - 1))
	mutatedCh = ch
	mutatedCh[r] = int(bool(mutatedCh[r]) ^ 1)
	return mutatedCh
			
"""Using breed and mutate it generates two new chromos from the selected pair"""
def breed (ch1, ch2):
	
	newCh1, newCh2 = [], []
	if random.random() < crossover_rate: # rate dependent crossover of selected chromosomes
		newCh1, newCh2 = crossover(ch1, ch2)
	else:
		newCh1, newCh2 = ch1, ch2
	newnewCh1 = mutate (newCh1) # mutate crossovered chromos
	newnewCh2 = mutate (newCh2)
	
	return newnewCh1, newnewCh2

""" Taking a ranked population return a new population by breeding the ranked one"""
def iteratePop (rankedPop):
	fitnessScores = [ item[-1] for item in rankedPop ] # extract fitness scores from ranked population
	rankedChromos = [ item[0] for item in rankedPop ] # extract chromosomes from ranked population

	newpop = []
	newpop.extend(rankedChromos[:7]) # known as elitism, conserve the best solutions to new population

	while len(newpop) != popN:
		ch1, ch2 = [], []
		ch1, ch2 = selectFittest (fitnessScores, rankedChromos) # select two of the fittest chromos
		
		ch1, ch2 = breed (ch1, ch2) # breed them to create two new chromosomes 
		newpop.append(ch1) # and append to new population
		newpop.append(ch2)
	return newpop

""""""
def embedHypotheses():
	good_length = len(good)
	bad_length = len(bad)

	temp = good[0]
	maxG = len(good[0][1])
	with open(LOG_DIR + 'good.csv', 'w') as f:
		for item in good:
			protein = item[1]
			if(len(protein) > maxG):
				maxG = len(protein)
			f.write(protein + '\n')

	with open(LOG_DIR + 'bad.csv', 'w') as f:
		for item in bad:
			protein = item[1]
			if(len(protein) > maxG):
				maxG = len(protein)
			f.write(protein + '\n')
	embedding_var_good = tf.Variable(tf.truncated_normal([good_length, 2]), name='embedding_good')
	embedding_var_bad = tf.Variable(tf.truncated_normal([bad_length, 2]), name='embedding_bad')
	with tf.Session() as sess:
		# Create summary writer.
		writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
		# Initialize embedding_var
		sess.run(embedding_var_good.initializer)
		sess.run(embedding_var_bad.initializer)
		# Create Projector config
		config_good = projector.ProjectorConfig()
		config_bad = projector.ProjectorConfig()
		# Add embedding visualizer
		embedding_good = config_good.embeddings.add()
		embedding_bad = config_bad.embeddings.add()
		# Attache the name 'embedding'
		embedding_good.tensor_name = 'embedding_good'
		embedding_bad.tensor_name = 'embedding_bad'
		# Metafile which is described later
		embedding_good.metadata_path = LOG_DIR + 'good.csv'
		embedding_bad.metadata_path = LOG_DIR + 'bad.csv'
		# Add writer and config to Projector
		projector.visualize_embeddings(writer, config_good)
		projector.visualize_embeddings(writer, config_bad)
		# Save the model
		saver_embed = tf.train.Saver([embedding_var_good, embedding_var_bad])
		saver_embed.save(sess, LOG_DIR + 'embedding_test.ckpt', 1)

		writer.close()


def main(): 
	#configureSettings ()
	chromos = generatePop() #generate new population of random chromosomes
	iterations = 0

	while iterations != max_iterations and solution_found != True:
		# take the pop of random chromos and rank them based on their fitness score/proximity to target output
		
		rankedPop = rankPop(chromos)
		tempRankedPop = copy.deepcopy(rankedPop)
		global good, bad
		# good.extend(rankedPop[:7])

		# good = itertools.islice(tempRankedPop, len(tempRankedPop) // 3)
		# bad = itertools.islice(tempRankedPop, len(tempRankedPop))
		good += tempRankedPop[:threshold]
		bad += tempRankedPop[threshold:]
		print("-------------")
		print(len(good))
		print("-------------")
		print(len(bad))

		# good = []
		# bad = []
		print ('\nCurrent iterations:', iterations)
		
		if solution_found != True:
			# if solution is not found iterate a new population from previous ranked population
			chromos = []
			chromos = iteratePop(rankedPop)
			iterations += 1
		else:
			break

	embedHypotheses()

if __name__ == "__main__":
	main()