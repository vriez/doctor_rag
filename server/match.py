CREATE (p1:Person {name: 'Albert Einstein'})
CREATE (t1:Theory {name: 'Theory of Relativity'})
CREATE (fos:FieldOfStudy {name: 'Physics'})

CREATE (p1)-[:DEVELOPED]->(t1)
CREATE (t1)-[:REVOLUTIONIZED]->(fos)


// Create nodes for entities
CREATE (vitaminDDeficiency:MedicalCondition {name: 'Vitamin D deficiency'})
CREATE (overlookedStatus:Status {name: 'overlooked'})
CREATE (covidSusceptibility:Propensity {name: 'susceptibility of COVID-19 infection'})
CREATE (covidSeverity:Degree {name: 'severity of COVID-19 infection'})

// Create relationships with properties
CREATE (vitaminDDeficiency)-[:IS_OFTEN]->(overlookedStatus)
CREATE (vitaminDDeficiency)-[:MIGHT_INFLUENCE]->(covidSusceptibility)
CREATE (vitaminDDeficiency)-[:MIGHT_INFLUENCE]->(covidSeverity)
