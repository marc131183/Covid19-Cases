# Covid19-Cases

## Filtering of data

The visualization tool comes with an option to filter data, before visualizing it, these are a few examples on how it works:

- Query 1: Get all countries that are in Asia  

      {Continent} = "Asia"

- Query 2: Get all countries that are either in Europe or have a population of 10000000 or bigger  

      {Continent} = "Europe" or {Population} >= 10000000

As one might have observed, columns should be put into curly brackets: {column_name}. Column names can be taken out of the visualization tool. Names should be put into quotation marks: "Name". Numbers can be put in without using any extra notation. And expressions should be linked by and/or.  
More examples:

- Query 3: Get all countries that are either in North- or South America and have a human development index of smaller than 30  

      ({Continent} = "North America" or {Continent} = "South America") and {Human_development_index} < 30

There are also operators that can be used on columns. These include mean, median, std, first, last, max, min. These operators can just be put in front of a column:

- Query 4: Get all countries that have an average number of new cases of 300 or higher and the maxmimum number of new deaths should be below 100  

      mean{New_cases} >= 300 and max{New_deaths} < 100

In addition we can also use the operators plus (+), minus (-), multiply (*) and divide (/). These can be put next to numerical values or columns that contain numerical values.

- Query 5: Get all rows of countries that have a value of new cases that is either below or higher than the mean +- 3*std (outliers)

      {New_cases} > mean{New_cases} + 3 * std{New_cases} or {New_cases} < mean{New_cases} - 3 * std{New_cases}
      
We can also add brackets, so that the expression looks more clear:

    ({New_cases} > (mean{New_cases} + 3 * std{New_cases})) or ({New_cases} < (mean{New_cases} - 3 * std{New_cases}))

