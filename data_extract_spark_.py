#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 21:47:51 2020

@author: joshi.purvi
""" 

import pyspark as ps
from  pyspark.sql.types import StructType,StructField,StringType

''' create spark session '''

spark = (ps.sql.SparkSession
         .builder
         .master('local[4]')
         .appName('lecture')
         .getOrCreate()
        )
sc = spark.sparkContext

from pyspark.sql import SQLContext
sqlcontext=SQLContext(sc)

def read_json():
    
    ''' json files can be downloaded from: https://www.yelp.com/dataset/challenge '''
    
    ''' make spark object of reading json files '''
    
    business=sqlcontext.read.json('yelp_dataset/business.json')
    review=sqlcontext.read.json('yelp_dataset/review.json')
    return business,review


def viewSchema(review,business):
    
    ''' to print schema '''
    
    review.printSchema()
    business.printSchema()

def create_temp_view(review,business):
    
    ''' carete temporory view on which select query will be run '''
    
    review.createOrReplaceTempView("yelp")
    business.createOrReplaceTempView("yelp_business")

def select_columns_query():
    
    ''' select and extract required columns from joining 2 tables review and busines '''
    
    new_buss = spark.sql("SELECT distinct b.business_id,b.name,b.categories,r.useful,r.text,r.stars,r.funny,r.cool from yelp_business b inner join yelp r on r.business_id = b.business_id where categories like '%India%'")
    return new_buss

def printTable(business_query):
    
    ''' print selected column's from tables review and business '''
    
    business_query.show()
    
def createPandasDataframe(business_query):
    
    ''' create pandas dataframe for spark sql dataframe '''
    
    panda_indian_buss_review = business_query.toPandas()
    return panda_indian_buss_review

def createCSV(panda_indian_review):
    
    ''' write selected columns from review and 'business tables '''
    
    panda_indian_review.to_csv("new_dataset_indian_resto_reiew.csv")
    
def main():
    
        ''' main function to call and execute all funtionality '''

    business,review=read_json()
    viewSchema(review,business)
    create_temp_view(review,business)
    business_query=select_columns_query()
    panda_indian_review= createPandasDataframe(business_query)
    createCSV(panda_indian_review)
    
if __name__ == "main":
    main()