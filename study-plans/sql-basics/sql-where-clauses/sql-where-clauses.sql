-- Write your SQL query here
SELECT name , salary
FROM employees
WHERE department IN ('Engineering', 'Marketing') and salary > 70000;
