import random

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax",base.Fitness,weights=(1.0,))#
creator.create("Individual",list,fitness=creator.FitnessMax)#個体の定義，遺伝子を保存する個体listに適応度fitnessというメンバ変数を追加

toolbox = base.Toolbox()

toolbox.register("attr_bool",random.randint,0,1)#遺伝子を作成する関数
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bool,100)#個体を作成する関数，100回toolbox.attr_hoolを実行して，その値をcreator.Individualに格納して返す関数
toolbox.register("population",tools.initRepeat,list,toolbox.individual)#個体をtoolbox.individualで作成し，listに格納し，世代を生成するpopulation関数

def evalOneMax(individual):
	return sum(individual),

toolbox.register("evaluate",evalOneMax)
toolbox.register("mate",tools.cxTwoPoint)
toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)
toolbox.register("select",tools.selTournament,tournsize=3)


def main():
	random.seed(64)
	pop = toolbox.population(n=300)#初期世代の作成，世代内の個体数はn=300.遺伝子，個体，世代を一気に作っている
	CXPB,MUTPB,NGEN = 0.5,0.2,40#交叉率，個体突然変異率，ループを回す世代数を指定

	print("Start of evolution")

	fitnesses = list(map(toolbox.evaluate,pop))#初期世代の個体の適応度を計算
	for ind,fit in zip(pop,fitnesses):#zip():複数のリストをまとめて習得
		ind.fitness.values = fit#固体内の適応度にfitness.valuesでアクセス

	print("Evaluated %i individuals" % len(pop))
	for g in range(NGEN):
		print("--Generation %i --" % g)

		offspring = toolbox.select(pop,len(pop))#選択した個体をoffspringに格納
		offspring = list(map(toolbox.clone,offspring))#その個体のクローンを作ってまた，offspringに格納

		for child1,child2 in zip(offspring[::2],offspring[1::2]):#offspringの偶数インデックスと奇数インデックスでそれぞれの個体を交叉
			if random.random() < CXPB: #CXPB:交叉率
				toolbox.mate(child1,child2)#交叉
				del child1.fitness.values#交叉後，適応度を再計算するため，delで削除
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < MUTPB: #突然変異率
				toolbox.mutate(mutant)
				del mutant.fitness.values

		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate,invalid_ind)
		for ind,fit in zip(invalid_ind,fitnesses):
			ind.fitness.values = fit

		print("Evaluated %i individuals" % len(invalid_ind))

		pop[:] = offspring

		fits = [ind.fitness.values[0] for ind in pop]

		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5

		print(" Min %s" % min(fits))
		print(" Max %s" % max(fits))
		print(" Avg %s" % mean)
		print(" Std %s" % std)
	print("-- End of (successful) evolution--")

	best_ind = tools.selBest(pop,1)[0]
	print("Best individual is %s,%s" % (best_ind,best_ind.fitness.values))

if __name__ == "__main__":
	main()



