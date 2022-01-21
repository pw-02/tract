use crate::frame::mmm::cost_model::CostModel;
pub fn models() -> Vec<(&'static str, CostModel)> {
vec!(
("generic_f32_4x4", CostModel { mr: 4, nr: 4,
intercept: 0.00003689411526838631,
coef: vec!(1.1892123225916299e-6, 2.1856630269339606e-8, 1.8007276957232266e-5, 5.053519861781259e-7, 2.714525273086468e-5, -1.9464266002347312e-7, 7.876053509501342e-6, 5.919391026987203e-6, -2.2346096284167092e-6, -1.097145356176169e-6, -1.5203718931911778e-5),
}),
("armv7neon_mmm_f32_8x4_cortexa7", CostModel { mr: 8, nr: 4,
intercept: 0.00010083460479096921,
coef: vec!(-2.3243383616535673e-6, 4.2219719094448326e-8, -3.414710416660876e-5, -8.749048496616462e-8, -5.631327074693315e-5, 1.403924359816871e-7, -6.181796583498462e-7, 2.479403326046156e-6, -6.421962498008331e-8, 1.9744397005264784e-7, 2.6355166337797464e-6),
}),
("armv7neon_mmm_f32_8x6_cortexa7", CostModel { mr: 8, nr: 6,
intercept: 0.000033884204795901825,
coef: vec!(-9.434066665983743e-7, 6.773489640128127e-8, 4.3787961762599356e-5, 1.9567133419719933e-7, 2.0360191927665707e-5, -4.846881220214299e-8, 1.4832676440116048e-6, -3.320423735164137e-8, 1.62645188017021e-7, 1.033463663395871e-6, -1.014573464640255e-5),
}),
("armv7neon_mmm_f32_8x4_cortexa9", CostModel { mr: 8, nr: 4,
intercept: 0.000050294403837273954,
coef: vec!(1.3580112104569158e-6, 8.9499284182424e-8, 1.6737519666345186e-5, -7.088503630985909e-7, 8.068967965838715e-6, 7.676967913668677e-7, 3.947350388095252e-6, -7.580193339042617e-6, -5.091888001892913e-7, -2.2397027262414596e-7, -5.249698365951377e-6),
}),
("armv7neon_mmm_f32_8x6_cortexa9", CostModel { mr: 8, nr: 6,
intercept: 0.00004534617808075629,
coef: vec!(-8.848862024985224e-7, 3.7361377202840784e-8, 1.8542136958919496e-5, 1.5991394756718872e-7, -2.1911165771904542e-5, 3.1554883903618124e-7, 7.36596883593276e-6, 1.1608565257667635e-5, -5.056111972038183e-7, -4.936607863908779e-7, -2.3066494615297598e-5),
}),
("armv7neon_mmm_f32_8x4_generic", CostModel { mr: 8, nr: 4,
intercept: 0.00009319668577180479,
coef: vec!(-3.5397285001704482e-6, 4.663677880702498e-8, -1.2311388977718626e-4, 8.948086696901887e-7, 6.895696037470718e-5, -8.067811628186185e-7, 3.0929028435663545e-6, 4.980721850068063e-6, -1.2835219097305232e-7, -2.2803283831931393e-6, -1.010773467811105e-6),
}),
("armv7neon_mmm_f32_8x6_generic", CostModel { mr: 8, nr: 6,
intercept: 0.00012621567756973732,
coef: vec!(-3.337810894017882e-6, 6.50660282841855e-8, -5.2056574896622336e-5, -5.090935708185159e-7, -1.0024202655315756e-4, 5.732944875372662e-7, -7.123223117688902e-7, 1.2784624647266214e-6, 3.8126390233692756e-7, 3.794940375173288e-7, 1.1043099158893268e-6),
}),
)}
