# Zarr DataFusion Query Optimizer Design (prompt)

Author: Alex Merose

Created: January 20, 2026

High level goal: we are working towards a first milestone, which is being able to replicate a benchmark in Extreme Weather Bench with SQL via zarr-datafusion (see https://github.com/jayendra13/zarr-datafusion/blob/main/freeze_evaluation_code_flow.md). To that end, we need to implement a SQL query optimizer. This optimizer needs to fit into our stack (Rust, DataFusion, Zarr with Xarray semantics). A core criterion for this query optimizer is that it needs to present a logical view of tables modeled by SQL but not violate the physical data structure semantics modeled by Xarray and Zarr. Please see the zarr-datafusion readme for an understanding of our model: https://github.com/jayendra13/zarr-datafusion/blob/main/README.md 

I propose that the core engine we use for our query optimizer is the egg library (which implements a form of equity saturation via e-graphs). Reference [2] explains how egg can be used for query optimizers in a database. 

Please read all of the links I provide. Then, read the zarr-datafusion sources to understand how it works. Then, propose a plan for how to build this query optimizer, making a mental model of how components work and relate to each other. Please write down notes as needed (design docs) in the repo in markdown. When testing, please write integration tests inspired by Extreme Weather Bench and the linked queries. 

Thanks so much, Claude. I’m very excited about this direction. 

## References 

Here are helpful references and some of my thoughts. References 1, 2, and 3 are checked in as PDFs in the design folder next to this README.

1. https://datafusion.apache.org/blog/2025/06/15/optimizing-sql-dataframes-part-two/ 
	- We can skip the pushdown optimizations and others for now. These will be implemented later. (Especially access path and join order selection, which seems hard.) For now, I want to focus on expression simplification, subquery rewrites, optimized expression evaluation, and using statistics directly.
	- Ideally, our optimizer should re-use or delegate to what datafusion provides out of the box as much as possible. 
	- It would be ideal to map metadata and statistics available from Xarray into datafusion affordances like ColumnStatistics and TableStatistics.
2. https://drops.dagstuhl.de/storage/00lipics/lipics-vol328-icdt2025/html/LIPIcs.ICDT.2025.34/LIPIcs.ICDT.2025.34.html 
3. https://docs.rs/egg/latest/egg/tutorials/_02_getting_started/index.html 
4. https://github.com/apache/datafusion/issues/1972#issuecomment-1069557904 
5. https://github.com/apache/datafusion/issues/1972#issuecomment-1077041378 
	- The “Database Theory in Action” [2] paper seems to provide solutions to the problems listed here: memorizing/caching the e graph data structure on a range of common patterns probably solves the slowness that the author pointed out here 
	- Xarray data variables are ordered by their dimensions (dim coordinates), so we naturally have statistical information we could use to make the optimizer fast. 
6.	https://github.com/apache/datafusion/issues/1972#issuecomment-1078838974 
	- I like this developer’s recommendation to use tunable parameters to choose where a query should fall in the pareto frontier involved in multi objective optimizations. 
7.	https://github.com/apache/datafusion/issues/1972#issuecomment-1154749123 
	- This solidifies my confidence that egg with datafusion will work well on xarray structured datasets. I think we can exploit the native statistical information made available in the xarray dataset model of Zarr. 
8. https://github.com/datafusion-contrib/datafusion-tokomak/tree/main/tokomak 
	- This is a previous attempt to build this integration that was never merged into datafusion. It’s four or five years old and both egg and DataFusion have improved since then. 
	- Implementation can be found in the src directory. The readme has a long todo list of features. 
	- In general, I wouldn’t trust these sources, but we can learn from this implementation. I think reference [2] solves the technical challenges found in these sources through search and caching.  

