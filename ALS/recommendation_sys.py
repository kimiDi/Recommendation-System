from pyspark.sql.types import *
from pyspark.sql import Row

link_sc = sc.textFile('ml-latest-small/links.csv').map(lambda line:line.split(","))
header = link_sc.first()
link_sc = link_sc.filter(lambda row： row != header)
l = link_sc.map(lambda p: Row(movieId = p[0], imdbId = p[1], tmdbId = p[2]))
schemalink = spark.createDataFrame(l)
schemalink.createOrReplaceTempView(‘link’)

spark.sql("select * from link").show()

movie_sc = sc.textFile('ml-latest-small/movies.csv').map(lambda line: line.split(","))
header = movie_sc.first()
movie_sc = movie_sc.filter(lambda row: row != header)
m = movie_sc.map(lambda p: Row(movieId = p[0], title = p[1], genres = p[2]))
schemamovie = spark.createDataFrame(m)
schemamovie.createOrReplaceTempView("movie")

spark.sql("select * from movie").show()

rate_sc = sc.textFile('ml-latest-small/ratings.csv').map(lambda line: line.split(","))
header = rate_sc.first()
rate_sc = rate_sc.filter(lambda row: row != header)
r = rate_sc.map(lambda p: Row(userId = p[0], movieId = p[1], rating = p[2], timestamp = p[3]))
schemarate = spark.createDataFrame(r)
schemarate.createOrReplaceTempView("rating")

spark.sql("select * from rating").show()

tag_sc = sc.textFile('ml-latest-small/tags.csv').map(lambda line: line.split(","))
header = tag_sc.first()
tag_sc = tag_sc.filter(lambda row: row != header)
t = tag_sc.map(lambda p : Row(userId = p[0], movieId = p[1], tag = p[2], timestamp = p[3]))
schematag = spark.createDataFrame(t)
schematag.createOrReplaceTempView('tag')

spark.sql("select * from tag").show()

# OLAP
# How many users
spark.sql("select count(distinct userId) as userNum from rating").show()

# How many movies
spark.sql("select count(distinct movieId) as movieNum from movie").show()

# How many movies are rated by users
spark.sql("select count(distinct movieId) as movieNum from rating").show()

# List movies which are not rated before
spark.sql("select title from movie where not exits (select * from rating where movie.movieId = rating.moiveId)").show()

# Spark ML ALS
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import time

rate_sc = sc.textFile('ml-latest-small/ratings.csv').map(lambda row: row.split(","))
header = rate_sc.first()
rate_sc = rate_sc.filter(lambda row: row != header)
ratings = rate_sc.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

ranks = [5, 10, 15, 20, 30]
numIterations = 10
reg_para - 0.1

train, validation = ratings.randomSplit([8, 2])
traindata = train.map(lambda p: (p[0], p[1]))
validata = validation.map(lambda p: (p[0], p[1]))

for rank in ranks:
	tic = time.clock()
	model = ALS.train(ratings, rank, iterations = numIterations, lambda_ = reg_para)
	predictions = model.predictAll(validata).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = validation.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	MSE = ratesAndPreds.map(lambda r:(r[1][0] - r[1][1])**2).mean()
	toc = time.clock()
	print("Rank {} Training MSE = {}".format(rank, MSE))
	print("Runtime {}".format(toc - tic))
