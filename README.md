# Project04

This project explores using the Random Forest Regression machine learning algorithm to predict renewable energy usage in the United States in a given year. 

This project's data comes from the [World Bank's World Development Indicators data portal](https://databank.worldbank.org/reports.aspx?source=2&series=EG.FEC.RNEW.ZS&country=). This data is free and can be downloaded directly via the data portal website, however my project accessed it using the World Bank's data API. 

This access does not require and API key, and my code uses the ["wbgapi"](https://pypi.org/project/wbgapi/) package for python to facilitate data base navigation and API requests. 

# Important Functions:
The first function in Final_Project.py is called 'getdata' and it takes one argument, which should be a list of search terms that are related to data variables you want to request. These requests are made within the function, using wbgapi's function 'wb.series.Series(q=...), or wb.series.info(q=...), with the q taking the search term as a string. 

The second important function is called columnlookup. This function takes two arguments: the searchterms list and the name of the directory where your project is stored locally. This function maps variable IDs from the data source to the descriptions of these variables, and outputs this mapping to a CSV file, which can be stored in the local directory of your choice. 
* Function calls are made in print statements in the 'if __name__ == "__main__" section of the file, but these functions can also be imported and called using a jupyter notebook. 
